import subprocess
import sys


def main():
    print("Hello from exp2!")
    scripts = []
    for p in ["1p/", "5p/", "10p/"]:
        path = "scripts/" + p
        for s in ["gpu_fit_models.py", "xgboost_train.py", "mlp_train.py", "independent_mlp.py", "shared_4class.py", "independent_4class.py"]:
            scripts.append(path + s)

    print("Scripts will be executed in the following order.")
    print(scripts)

    for script in scripts:
        print(f"Starting: {script}")

        result = subprocess.run([sys.executable, script], capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.returncode != 0:
            print(f"Error occured in {script}:")
            print(result.stderr)
            sys.exit(result.returncode)

    print("All scripts executed")



if __name__ == "__main__":
    main()
