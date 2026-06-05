import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.runtime.sandbox_mgr import DockerSandbox

def run_docker_sandbox_test():
    load_dotenv()

    project_root = Path(__file__).resolve().parent
    repo_path = project_root / "data" / "clones" / "juice-shop"

    if not repo_path.exists():
        print(f"❌ Error: Juice Shop codebase not found at {repo_path}")
        print("Please ensure the repository is cloned before running this test.")
        return

    print("==================================================")
    print("STARTING DOCKER SANDBOX INTEGRATION TEST ON JUICE SHOP")
    print("==================================================")
    print(f"Repo Path: {repo_path}")
    print(f"Docker Image: node:20-alpine\n")

    # 1. Start sandbox
    print("--- 1️⃣ Starting Ephemeral Sandbox ---")
    try:
        sandbox = DockerSandbox(repo_path, image="node:20-alpine")
        sandbox.start()
        print("✅ Sandbox started successfully!")
        print(f"   Container Name: {sandbox._container_name}")
    except RuntimeError as e:
        print(f"❌ Failed to start Sandbox: {e}")
        print("Please ensure Docker Desktop is running and that your user has permission to access the Docker daemon.")
        return
    except Exception as e:
        print(f"❌ Unexpected error starting Sandbox: {e}")
        return

    try:
        # 2. Check environment inside sandbox
        print("\n--- 2️⃣ Checking Environment inside Sandbox ---")
        node_res = sandbox.run("node --version")
        print(f"   Node version command exit code: {node_res.exit_code}")
        print(f"   Node version stdout: {node_res.stdout.strip()}")
        
        npm_res = sandbox.run("npm --version")
        print(f"   NPM version command exit code: {npm_res.exit_code}")
        print(f"   NPM version stdout: {npm_res.stdout.strip()}")

        # 3. Verify copied repository structure
        print("\n--- 3️⃣ Verifying Repository Copied to /workspace ---")
        ls_res = sandbox.run("ls -lh /workspace")
        print(f"   ls exit code: {ls_res.exit_code}")
        print("   ls output:")
        for line in ls_res.stdout.splitlines()[:15]:
            print(f"     {line}")
        if len(ls_res.stdout.splitlines()) > 15:
            print("     ...")

        # 4. File I/O: Read file inside container
        print("\n--- 4️⃣ File I/O: Reading package.json from Sandbox ---")
        pkg_content = sandbox.read_file("package.json")
        if pkg_content:
            try:
                pkg_data = json.loads(pkg_content)
                print(f"   Successfully read package.json!")
                print(f"   Name: {pkg_data.get('name')}")
                print(f"   Version: {pkg_data.get('version')}")
                print(f"   Dependencies count: {len(pkg_data.get('dependencies', {}))}")
            except Exception as e:
                print(f"   ❌ Failed to parse package.json JSON: {e}")
        else:
            print("   ❌ Failed to read package.json from sandbox!")

        # 5. File I/O: Write new file inside container
        print("\n--- 5️⃣ File I/O: Writing and Reading a custom file ---")
        test_file_path = "sandbox_integration_test.txt"
        test_file_content = "This is a temporary file written by the DockerSandbox integration test."
        print(f"   Writing file: {test_file_path}")
        sandbox.write_file(test_file_path, test_file_content)
        
        read_back_content = sandbox.read_file(test_file_path)
        if read_back_content == test_file_content:
            print("   ✅ File read/write verification succeeded!")
        else:
            print(f"   ❌ Read content mismatch! Expected: {test_file_content!r}, Got: {read_back_content!r}")

        # 6. File I/O & Sandbox Security: Modify package.json and verify isolation
        print("\n--- 6️⃣ Modifying package.json in Sandbox (Verifying Isolation) ---")
        if pkg_content:
            try:
                pkg_data = json.loads(pkg_content)
                pkg_data["version"] = "99.9.9-sandbox-test"
                pkg_data["dependencies"]["antigravity-dummy-pkg"] = "1.0.0"
                
                print("   Writing modified package.json to sandbox...")
                sandbox.write_file("package.json", json.dumps(pkg_data, indent=2))
                
                # Check sandbox file
                modified_sandbox_pkg = sandbox.read_file("package.json")
                sandbox_pkg_data = json.loads(modified_sandbox_pkg)
                print(f"   Sandbox package version: {sandbox_pkg_data.get('version')}")
                print(f"   Sandbox dependency added: {'antigravity-dummy-pkg' in sandbox_pkg_data.get('dependencies', {})}")
                
                # Check host file (should be unchanged!)
                with open(repo_path / "package.json", "r", encoding="utf-8") as f:
                    host_pkg_data = json.load(f)
                
                print(f"   Host package version: {host_pkg_data.get('version')}")
                print(f"   Host dependency present: {'antigravity-dummy-pkg' in host_pkg_data.get('dependencies', {})}")
                
                if host_pkg_data.get("version") == "20.0.0" and "antigravity-dummy-pkg" not in host_pkg_data.get("dependencies", {}):
                    print("   ✅ Host isolation verified! Host filesystem was NOT touched.")
                else:
                    print("   ❌ Host files were modified! Isolation failed!")
            except Exception as e:
                print(f"   ❌ Error verifying isolation: {e}")

        # 7. Run failing command
        print("\n--- 7️⃣ Running a command that fails (returns non-zero) ---")
        fail_res = sandbox.run("false")
        print(f"   exit_code: {fail_res.exit_code}")
        print(f"   stderr: {fail_res.stderr.strip() or '(none)'}")
        if fail_res.exit_code != 0:
            print("   ✅ Correctly captured failure exit code!")
        else:
            print("   ❌ Error: Command 'false' returned exit code 0!")

        # 8. Run command using node in sandbox
        print("\n--- 8️⃣ Running a quick Node.js script inside the sandbox ---")
        node_script_res = sandbox.run("node -e \"console.log('Hello from sandbox execution node script'); process.exit(0);\"")
        print(f"   exit_code: {node_script_res.exit_code}")
        print(f"   stdout: {node_script_res.stdout.strip()}")

    finally:
        # 9. Tearing Down Sandbox
        print("\n--- 9️⃣ Tearing Down Sandbox ---")
        sandbox.teardown()
        print("✅ Sandbox torn down and container removed.")

    # 10. Verify Timeout handling
    print("\n--- 🔟 Testing Command Timeout Handling ---")
    try:
        with DockerSandbox(repo_path, image="node:20-alpine") as sb_timeout:
            print("   Starting timeout sandbox container...")
            # We run a sleep for 10 seconds, but set a timeout of 2 seconds
            timeout_res = sb_timeout.run("sleep 10", timeout=2)
            print(f"   Command duration: {timeout_res.duration_seconds}s")
            print(f"   Command exit_code: {timeout_res.exit_code}")
            print(f"   Command stderr: {timeout_res.stderr.strip()}")
            if timeout_res.exit_code == 124:
                print("   ✅ Timeout handled correctly! (Exit code 124)")
            else:
                print(f"   ❌ Unexpected exit code for timeout: {timeout_res.exit_code}")
    except Exception as e:
        print(f"   ❌ Error testing timeout: {e}")

    print("\n==================================================")
    print("DOCKER SANDBOX INTEGRATION TEST COMPLETED")
    print("==================================================")

if __name__ == "__main__":
    run_docker_sandbox_test()
