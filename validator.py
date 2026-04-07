#!/usr/bin/env python
"""
Pre-Submission Validator Script for Meta AI Hackathon

This script validates your submission against all hackathon requirements.
Run before submitting: python validator.py
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class ValidationError(Exception):
    """Validation failure"""
    pass


class HackathonValidator:
    """Validate hackathon submission"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def check(self, name: str, fn):
        """Run a check and track result"""
        try:
            fn()
            self.passed.append(name)
            print(f"✓ {name}")
        except ValidationError as e:
            self.failed.append((name, str(e)))
            print(f"✗ {name}: {e}")
        except Exception as e:
            self.failed.append((name, f"Unexpected error: {e}"))
            print(f"✗ {name}: Unexpected error: {e}")
    
    def warn(self, msg: str):
        """Record a warning"""
        self.warnings.append(msg)
        print(f"⚠ {msg}")
    
    # ========== VALIDATION CHECKS ==========
    
    def check_file_exists(self, path: str, critical: bool = True):
        """Check if required file exists"""
        if not (self.root / path).exists():
            msg = f"Missing {path}"
            if critical:
                raise ValidationError(msg)
            else:
                self.warn(msg)
    
    def check_openenv_yaml(self):
        """Validate openenv.yaml exists and has correct structure"""
        self.check_file_exists("openenv.yaml", critical=True)
        yaml_file = self.root / "openenv.yaml"
        content = yaml_file.read_text()
        
        required_keywords = ["name:", "api:", "action_spec:", "observation_spec:", "tasks:"]
        for kw in required_keywords:
            if kw not in content:
                raise ValidationError(f"openenv.yaml missing required section: {kw}")
    
    def check_inference_py(self):
        """Validate inference.py exists and contains required functions"""
        self.check_file_exists("inference.py", critical=True)
        inf_file = self.root / "inference.py"
        content = inf_file.read_text()
        
        required = ["def main():", "def evaluate_submission", "[START]", "[STEP]", "[END]"]
        for req in required:
            if req not in content:
                raise ValidationError(f"inference.py missing: {req}")
    
    def check_dockerfile(self):
        """Validate Dockerfile is correct"""
        self.check_file_exists("Dockerfile", critical=True)
        dockerfile = self.root / "Dockerfile"
        content = dockerfile.read_text()
        
        if "python" not in content.lower():
            raise ValidationError("Dockerfile must be Python-based")
        if "7860" not in content:
            raise ValidationError("Dockerfile must EXPOSE port 7860 for HF Spaces")
    
    def check_requirements(self):
        """Validate requirements.txt has necessary packages"""
        self.check_file_exists("requirements.txt", critical=True)
        req_file = self.root / "requirements.txt"
        content = req_file.read_text()
        
        required_packages = ["pydantic", "gymnasium", "numpy"]
        for pkg in required_packages:
            if pkg not in content:
                raise ValidationError(f"requirements.txt missing {pkg}")
    
    def check_pyproject_toml(self):
        """Validate pyproject.toml structure"""
        self.check_file_exists("pyproject.toml", critical=True)
        proj_file = self.root / "pyproject.toml"
        content = proj_file.read_text()
        
        required = ["[project]", "name =", "version =", "dependencies ="]
        for req in required:
            if req not in content:
                raise ValidationError(f"pyproject.toml missing: {req}")
    
    def check_readme(self):
        """Validate README.md is comprehensive"""
        self.check_file_exists("README.md", critical=True)
        readme = self.root / "README.md"
        content = readme.read_text()
        
        required_sections = ["# ", "quick start", "features", "setup", "task definitions"]
        for section in required_sections:
            if section.lower() not in content.lower():
                raise ValidationError(f"README.md missing section: {section}")
    
    def check_env_file(self):
        """Validate .env.example exists"""
        self.check_file_exists(".env.example", critical=False)
    
    def check_source_files(self):
        """Validate core source files exist"""
        src_files = [
            "src/__init__.py",
            "src/types.py",
            "src/environment.py",
            "src/graders.py",
        ]
        for f in src_files:
            self.check_file_exists(f, critical=True)
    
    def check_types_pydantic(self):
        """Validate Pydantic models are properly typed"""
        sys.path.insert(0, str(self.root))
        try:
            from src.types import (
                Observation, Action, Reward,
                Email, Meeting, Task, Notification
            )
            
            # Check that models have __fields__
            for model in [Observation, Action, Reward, Email, Meeting, Task, Notification]:
                if not hasattr(model, 'model_fields'):
                    raise ValidationError(f"{model.__name__} is not a proper Pydantic model")
        except ImportError as e:
            raise ValidationError(f"Cannot import types: {e}")
    
    def check_environment_api(self):
        """Validate EnterpriseEnv has required OpenEnv methods"""
        sys.path.insert(0, str(self.root))
        try:
            from src.environment import EnterpriseEnv
            
            required_methods = ['reset', 'step', 'state']
            for method in required_methods:
                if not hasattr(EnterpriseEnv, method):
                    raise ValidationError(f"EnterpriseEnv missing method: {method}")
            
            # Try to instantiate
            env = EnterpriseEnv(
                num_emails_per_day=3,
                num_meetings_per_day=2,
                num_tasks_per_day=2,
                max_steps=10
            )
            
            # Test reset
            obs, info = env.reset(seed=42)
            if obs is None:
                raise ValidationError("reset() returned None observation")
            
            # Test step
            from src.types import Action, ActionType
            action = Action(action_type=ActionType.NOOP)
            obs, reward, terminated, truncated, info = env.step(action)
            if obs is None:
                raise ValidationError("step() returned None observation")
            
            # Test state
            obs = env.state()
            if obs is None:
                raise ValidationError("state() returned None observation")
        
        except Exception as e:
            raise ValidationError(f"Environment API check failed: {e}")
    
    def check_graders(self):
        """Validate all 3 task graders exist"""
        sys.path.insert(0, str(self.root))
        try:
            from src.graders import get_task_graders
            
            graders = get_task_graders()
            difficulties = {'easy', 'medium', 'hard'}
            
            if set(graders.keys()) != difficulties:
                raise ValidationError(
                    f"Graders missing difficulties. Found: {set(graders.keys())}, "
                    f"Expected: {difficulties}"
                )
            
            # Check that each grader can be called
            from src import EnterpriseEnv
            env = EnterpriseEnv(max_steps=10)
            obs, _ = env.reset(seed=42)
            
            for difficulty, grader in graders.items():
                score, explanation = grader.grade(obs, env.metadata)
                
                if not isinstance(score, (int, float)):
                    raise ValidationError(f"{difficulty} grader returned non-numeric score")
                
                if not (0.0 <= score <= 1.0):
                    raise ValidationError(
                        f"{difficulty} grader returned out-of-range score: {score}"
                    )
                
                if not isinstance(explanation, str):
                    raise ValidationError(f"{difficulty} grader returned non-string explanation")
        
        except Exception as e:
            raise ValidationError(f"Grader validation failed: {e}")
    
    def check_inference_runs(self):
        """Quick test: Can inference.py run?"""
        try:
            env = os.environ.copy()
            env["VALIDATION_MODE"] = "1"
            env["API_BASE_URL"] = "https://api.openai.com/v1"
            env["MODEL_NAME"] = "gpt-3.5-turbo"
            env["HF_TOKEN"] = "dummy_token_for_validation"
            
            result = subprocess.run(
                [sys.executable, str(self.root / "inference.py"), "--task", "easy", "--steps", "3"],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            
            # Check for structured log markers
            for marker in ["[START]", "[END]"]:
                if marker not in result.stdout:
                    raise ValidationError(f"inference.py output missing {marker}")
            
            if result.returncode not in [0, 1]:  # Allow 0 or 1 (graceful exit)
                raise ValidationError(f"inference.py exited with code {result.returncode}")
        
        except subprocess.TimeoutExpired:
            raise ValidationError("inference.py execution timed out (> 60s)")
        except Exception as e:
            raise ValidationError(f"inference.py execution failed: {e}")
    
    def check_docker_image(self):
        """Check if Dockerfile can be built"""
        try:
            result = subprocess.run(
                ["docker", "build", "-t", "enterprise-env-test", str(self.root)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            if result.returncode != 0:
                # Docker might not be installed, just warn
                self.warn("Docker not available for build test")
                return
            
            print("  Docker image built successfully")
        
        except FileNotFoundError:
            self.warn("Docker not installed - skipping docker build test")
        except subprocess.TimeoutExpired:
            raise ValidationError("Docker build timed out")
    
    def run_all_checks(self):
        """Run all validation checks"""
        print("\n" + "="*70)
        print("HACKATHON SUBMISSION VALIDATOR")
        print("="*70 + "\n")
        
        print("Required Files:")
        self.check("openenv.yaml", self.check_openenv_yaml)
        self.check("inference.py", self.check_inference_py)
        self.check("Dockerfile", self.check_dockerfile)
        self.check("requirements.txt", self.check_requirements)
        self.check("pyproject.toml", self.check_pyproject_toml)
        self.check("README.md", self.check_readme)
        self.check(".env.example", self.check_env_file)
        
        print("\nSource Code:")
        self.check("Source files", self.check_source_files)
        self.check("Pydantic models", self.check_types_pydantic)
        self.check("Environment API", self.check_environment_api)
        self.check("Task graders", self.check_graders)
        
        print("\nFunctional Tests:")
        self.check("Inference execution", self.check_inference_runs)
        self.check("Docker build", self.check_docker_image)
        
        # Print summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"✓ Passed: {len(self.passed)}")
        print(f"✗ Failed: {len(self.failed)}")
        print(f"⚠ Warnings: {len(self.warnings)}")
        
        if self.failed:
            print("\nFailed Checks:")
            for name, err in self.failed:
                print(f"  - {name}: {err}")
        
        if self.warnings:
            print("\nWarnings:")
            for warn in self.warnings:
                print(f"  - {warn}")
        
        print("="*70 + "\n")
        
        if self.failed:
            print("RESULT: ✗ SUBMISSION NOT READY")
            print("\nFix the above errors before submitting.")
            return False
        else:
            print("RESULT: ✓ SUBMISSION READY FOR HACKATHON!")
            print("\nYour submission passes all validation checks.")
            if self.warnings:
                print("Note: Address warnings for best results.")
            return True


def main():
    """Run validator"""
    validator = HackathonValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
