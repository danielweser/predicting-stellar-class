import subprocess
import sys

def execute_module(script_name):
    print(f"\n[{script_name}]")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Execution failed in module {script_name} with exit code {e.returncode}.")
        sys.exit(1)

def main():
    print("Executing training pipeline...")
    
    modules = [
        "train_xgb.py",
        "train_mlp.py",
        "optimize_blend.py"
    ]
    
    for module in modules:
        execute_module(module)
        
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()