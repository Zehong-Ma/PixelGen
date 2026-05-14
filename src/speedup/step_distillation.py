"""
Step Distillation for PixelGen.

Trains a student model to match teacher predictions in fewer steps.
This is a form of knowledge distillation where the student learns
to "skip" intermediate denoising steps.

Key idea: Train student to go from x_t to x_{t-k} in one step,
where teacher takes k steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """Configuration for step distillation."""
    teacher_steps: int = 50      # Teacher uses many steps
    student_steps: int = 4       # Student learns to match in fewer
    distill_loss_weight: float = 1.0
    feature_loss_weight: float = 0.1  # Optional intermediate feature matching
    ema_decay: float = 0.9999    # EMA for teacher (if using self-distillation)


class StepDistillationTrainer:
    """
    Trains student to match teacher's multi-step denoising in fewer steps.

    Strategy:
    1. Teacher does k steps: x_t -> x_{t-1} -> ... -> x_{t-k}
    2. Student learns: x_t -> x_{t-k} in one step
    3. Loss = ||student(x_t, t) - teacher_final||^2

    This allows the student to learn "trajectory shortcuts".
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig = None,
    ):
        """
        Initialize distillation trainer.

        Args:
            teacher: Pre-trained teacher model (frozen)
            student: Student model to train (can start from teacher weights)
            config: DistillationConfig
        """
        self.config = config or DistillationConfig()
        self.teacher = teacher
        self.student = student

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Compute step ratio
        self.step_ratio = self.config.teacher_steps // self.config.student_steps

    def get_teacher_target(
        self,
        x_t: torch.Tensor,
        t_start: float,
        t_end: float,
        condition: torch.Tensor,
        num_teacher_steps: int,
    ) -> torch.Tensor:
        """
        Run teacher for multiple steps to get target.

        Args:
            x_t: Noisy input at timestep t_start
            t_start: Starting timestep
            t_end: Ending timestep
            condition: Class labels
            num_teacher_steps: Number of teacher steps to take

        Returns:
            Teacher's output after num_teacher_steps
        """
        device = x_t.device
        batch_size = x_t.shape[0]

        x = x_t.clone()
        timesteps = torch.linspace(t_start, t_end, num_teacher_steps + 1, device=device)

        with torch.no_grad():
            for i in range(num_teacher_steps):
                t_cur = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_cur

                t_batch = t_cur.expand(batch_size)

                # Teacher prediction (x-prediction in flow matching)
                pred = self.teacher(x, t_batch, condition)

                # Compute velocity and step
                v = (pred - x) / (1 - t_cur).clamp_min(0.002)
                x = x + v * dt

        return x

    def compute_distillation_loss(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss for one training step.

        Student tries to match teacher's k-step output in one step.
        """
        batch_size = x_t.shape[0]
        device = x_t.device

        # Determine step span for student
        dt = 1.0 / self.config.student_steps
        t_start = t
        t_end = (t + dt).clamp_max(0.998)

        # Get teacher target (multiple steps)
        teacher_steps_per_student = max(1, self.step_ratio)
        teacher_target = self.get_teacher_target(
            x_t, t_start.mean().item(), t_end.mean().item(),
            condition, teacher_steps_per_student
        )

        # Student prediction (single step)
        student_pred = self.student(x_t, t, condition)

        # MSE loss
        loss = F.mse_loss(student_pred, teacher_target)

        return loss * self.config.distill_loss_weight

    def train_step(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        Single training step for distillation.

        Args:
            x_0: Clean images [B, 3, H, W]
            condition: Class labels [B]
            optimizer: Optimizer for student

        Returns:
            Dict with loss values
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timestep
        t = torch.rand(batch_size, device=device) * 0.996 + 0.002

        # Create noisy sample (flow matching interpolation)
        noise = torch.randn_like(x_0)
        x_t = t.view(-1, 1, 1, 1) * x_0 + (1 - t.view(-1, 1, 1, 1)) * noise

        # Compute loss
        optimizer.zero_grad()
        loss = self.compute_distillation_loss(x_t, t, condition)
        loss.backward()
        optimizer.step()

        return {'distill_loss': loss.item()}


class ProgressiveDistillation:
    """
    Progressive distillation: repeatedly halve the number of steps.

    Process:
    1. Start with N-step teacher
    2. Train N/2-step student to match teacher
    3. Student becomes new teacher
    4. Train N/4-step student
    5. Repeat until desired step count

    This gradual reduction is more stable than direct distillation.
    """

    def __init__(
        self,
        initial_teacher: nn.Module,
        initial_steps: int = 64,
        final_steps: int = 4,
    ):
        self.initial_teacher = initial_teacher
        self.initial_steps = initial_steps
        self.final_steps = final_steps

        # Calculate number of distillation rounds
        self.num_rounds = 0
        steps = initial_steps
        while steps > final_steps:
            steps = steps // 2
            self.num_rounds += 1

        print(f"[ProgressiveDistillation] {self.num_rounds} rounds: "
              f"{initial_steps} -> {final_steps} steps")

    def get_round_config(self, round_idx: int) -> Tuple[int, int]:
        """Get teacher/student steps for a given round."""
        teacher_steps = self.initial_steps // (2 ** round_idx)
        student_steps = teacher_steps // 2
        return teacher_steps, student_steps

    def run_round(
        self,
        teacher: nn.Module,
        student: nn.Module,
        dataloader,
        num_epochs: int,
        teacher_steps: int,
        student_steps: int,
    ):
        """
        Run one round of progressive distillation.

        After this round, student can match teacher quality in half the steps.
        """
        config = DistillationConfig(
            teacher_steps=teacher_steps,
            student_steps=student_steps,
        )

        trainer = StepDistillationTrainer(teacher, student, config)
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)

        print(f"  Training {student_steps}-step student to match {teacher_steps}-step teacher")

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in dataloader:
                x_0, condition, _ = batch
                losses = trainer.train_step(x_0, condition, optimizer)
                epoch_loss += losses['distill_loss']
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

        return student


class ConsistencyDistillation:
    """
    Consistency distillation for single-step generation.

    Trains model to satisfy consistency property:
    f(x_t, t) = f(x_{t'}, t') for any t, t' on same trajectory

    This enables single-step generation by mapping any x_t directly to x_0.
    """

    def __init__(
        self,
        model: nn.Module,
        ema_decay: float = 0.9999,
    ):
        self.model = model
        self.ema_decay = ema_decay

        # Create EMA model (target)
        self.ema_model = self._create_ema()

    def _create_ema(self) -> nn.Module:
        """Create EMA copy of model."""
        import copy
        ema = copy.deepcopy(self.model)
        for param in ema.parameters():
            param.requires_grad = False
        return ema

    def _update_ema(self):
        """Update EMA parameters."""
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(),
                                         self.ema_model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )

    def consistency_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency loss.

        Key insight: Model output should be consistent across timesteps
        on the same ODE trajectory.
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample two nearby timesteps
        t1 = torch.rand(batch_size, device=device) * 0.8 + 0.1
        t2 = t1 + torch.rand(batch_size, device=device) * 0.1
        t2 = t2.clamp_max(0.99)

        # Create noisy samples on same trajectory
        noise = torch.randn_like(x_0)
        x_t1 = t1.view(-1, 1, 1, 1) * x_0 + (1 - t1.view(-1, 1, 1, 1)) * noise
        x_t2 = t2.view(-1, 1, 1, 1) * x_0 + (1 - t2.view(-1, 1, 1, 1)) * noise

        # Model predictions (should be consistent)
        pred1 = self.model(x_t1, t1, condition)

        with torch.no_grad():
            pred2 = self.ema_model(x_t2, t2, condition)

        # Consistency loss
        loss = F.mse_loss(pred1, pred2)

        return loss

    def train_step(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> dict:
        """Single training step."""
        optimizer.zero_grad()
        loss = self.consistency_loss(x_0, condition)
        loss.backward()
        optimizer.step()

        self._update_ema()

        return {'consistency_loss': loss.item()}
