#!/usr/bin/env python3
"""
Demo script for BuilderBrain Hugging Face Integration.

Shows how to export models, run inference, and deploy to HF Hub.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'transformers', 'datasets', 'accelerate', 'wandb', 'gradio', 'huggingface-hub'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing with uv...")
        subprocess.run(['uv', 'add'] + missing_packages, check=True)
        return True

    return False


def demo_model_export():
    """Demo model export functionality."""
    print("📦 Model Export Demo")
    print("=" * 50)

    try:
        from huggingface_pipeline.model_export.export import ModelExporter

        exporter = ModelExporter()

        print("🔄 Exporting BuilderBrain model to HF format...")
        result = exporter.export_to_huggingface(
            model_path="builderbrain_final.ckpt",
            config_path="configs/small.yaml",
            scale="small"
        )

        if "error" not in result:
            print(f"✅ Export completed: {result['export_id']}")
            print(f"📁 Export path: {result['export_path']}")
            print(f"💾 File size: {result['file_size']}")

            return result['export_path']
        else:
            print(f"❌ Export failed: {result['error']}")
            return None

    except Exception as e:
        print(f"❌ Export demo failed: {e}")
        return None


def demo_training_integration():
    """Demo HF Trainer integration."""
    print("\n🎯 Training Integration Demo")
    print("=" * 50)

    try:
        from huggingface_pipeline.training_integration.hf_trainer import main as train_main

        print("🔄 Running BuilderBrain training with HF integration...")
        print("(This is a demo - training will be quick)")

        # Run a quick training demo
        subprocess.run([
            sys.executable, '-c',
            '''
import sys
sys.path.append(".")
from huggingface_pipeline.training_integration.hf_trainer import main
main()
            '''
        ], timeout=60)  # Timeout after 60 seconds

        print("✅ Training demo completed")

    except subprocess.TimeoutExpired:
        print("⏱️ Training demo timed out (expected for demo)")
    except Exception as e:
        print(f"❌ Training demo failed: {e}")


def demo_api_inference():
    """Demo API inference functionality."""
    print("\n🧠 API Inference Demo")
    print("=" * 50)

    # Start API server if not already running
    api_url = "http://localhost:8000"

    try:
        # Test health endpoint
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("❌ API server health check failed")
            return

    except requests.exceptions.ConnectionError:
        print("❌ API server is not running")
        print("💡 Start the API server first: python -m huggingface_pipeline.inference_api.app")
        return

    # Test inference endpoints
    test_prompts = [
        "Generate a JSON API call for user authentication",
        "Create a robot manipulation plan for picking and placing",
        "Design a phone conversation flow for customer support"
    ]

    for prompt in test_prompts:
        print(f"\n🔄 Testing prompt: {prompt[:50]}...")

        try:
            response = requests.post(
                f"{api_url}/inference/generate",
                json={
                    "prompt": prompt,
                    "model_scale": "small",
                    "grammar_strict": True,
                    "max_tokens": 100
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {result['response'][:100]}...")
                print(f"   Tokens: {result['tokens_generated']}, Time: {result['processing_time']:.2f}s")
            else:
                print(f"❌ Inference failed: {response.status_code}")

        except Exception as e:
            print(f"❌ Request failed: {e}")


def demo_grammar_validation():
    """Demo grammar validation functionality."""
    print("\n🔤 Grammar Validation Demo")
    print("=" * 50)

    api_url = "http://localhost:8000"

    # Test grammar constraints endpoint
    try:
        response = requests.get(f"{api_url}/grammar/constraints", timeout=5)

        if response.status_code == 200:
            constraints = response.json()
            print("✅ Available grammars:")
            for grammar in constraints['available_grammars']:
                print(f"   • {grammar}")

            # Test grammar preview
            test_text = '{"name": "test", "value": 123}'
            preview_response = requests.post(
                f"{api_url}/grammar/preview",
                json={"text": test_text, "grammar_type": "json"},
                timeout=5
            )

            if preview_response.status_code == 200:
                preview = preview_response.json()
                print(f"✅ Grammar preview for: {test_text}")
                print(f"   Violations: {len(preview.get('violations', []))}")
            else:
                print("❌ Grammar preview failed")
        else:
            print("❌ Grammar constraints check failed")

    except requests.exceptions.ConnectionError:
        print("❌ API server not running for grammar demo")
    except Exception as e:
        print(f"❌ Grammar demo failed: {e}")


def demo_plan_validation():
    """Demo plan validation functionality."""
    print("\n🔧 Plan Validation Demo")
    print("=" * 50)

    api_url = "http://localhost:8000"

    # Test plan validation
    test_plan = {
        "nodes": [
            {"id": "grasp", "type": "grasp", "params": {"object_id": "red_cube"}},
            {"id": "rotate", "type": "rotate", "params": {"angle": 90}},
            {"id": "place", "type": "place", "params": {"target_pose": {"x": 0.5, "y": 0.2}}}
        ],
        "edges": [
            {"from": "grasp", "to": "rotate", "type": "seq"},
            {"from": "rotate", "to": "place", "type": "seq"}
        ]
    }

    try:
        response = requests.post(
            f"{api_url}/plans/validate",
            json=test_plan,
            timeout=5
        )

        if response.status_code == 200:
            validation = response.json()
            print(f"✅ Plan validation: {'Valid' if validation['valid'] else 'Invalid'}")
            print(f"   Validation time: {validation['validation_time']:.3f}s")
            print(f"   Errors: {len(validation['errors'])}")
            print(f"   Warnings: {len(validation['warnings'])}")
        else:
            print("❌ Plan validation failed")

        # Test execution preview
        preview_response = requests.post(
            f"{api_url}/plans/preview",
            json=test_plan,
            timeout=5
        )

        if preview_response.status_code == 200:
            preview = preview_response.json()
            print(f"✅ Execution preview: {preview['estimated_execution_time']:.2f}s")
            print(f"   Risk assessment: {preview['risk_assessment']}")
        else:
            print("❌ Execution preview failed")

    except requests.exceptions.ConnectionError:
        print("❌ API server not running for plan demo")
    except Exception as e:
        print(f"❌ Plan demo failed: {e}")


def demo_deployment():
    """Demo deployment functionality."""
    print("\n🚀 Deployment Demo")
    print("=" * 50)

    print("🐳 Docker Deployment:")
    print("   docker-compose up -d")
    print("   Access dashboard at: http://localhost:8501")
    print("   Access API docs at: http://localhost:8000/docs")

    print("\n☸️ Kubernetes Deployment:")
    print("   kubectl apply -f huggingface_pipeline/deployment/k8s/")
    print("   Access via ingress at: http://builderbrain.local")

    print("\n🤗 Hugging Face Integration:")
    print("   Export models: python -m huggingface_pipeline.model_export.export")
    print("   Upload to Hub: python -m huggingface_pipeline.model_export.upload_to_hub")


def main():
    """Main demo function."""
    print("🤗 BuilderBrain Hugging Face Integration Demo")
    print("=" * 60)

    # Check dependencies
    if check_dependencies():
        print("Dependencies installed. Please run the demo again.")
        return

    # Run demo sections
    demo_model_export()
    demo_api_inference()
    demo_grammar_validation()
    demo_plan_validation()
    demo_training_integration()
    demo_deployment()

    print("\n🎉 Demo completed!")
    print("\n📚 Next Steps:")
    print("   1. Set up Hugging Face token for model uploads")
    print("   2. Configure production deployment settings")
    print("   3. Run full training with real datasets")
    print("   4. Deploy to production with monitoring")


if __name__ == "__main__":
    main()
