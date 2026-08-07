import os
import shutil
from src.tools.file_ops.handler import FileOpsHandler
from src.tools.file_ops.permissions import FilePermissions

def auto_ask_callback(prompt_msg: str) -> bool:
    print(f"[TEST CALLBACK] ECHELON asks: {prompt_msg}")
    print("[TEST CALLBACK] Auto-replying YES")
    return True

def auto_deny_callback(prompt_msg: str) -> bool:
    print(f"[TEST CALLBACK] ECHELON asks: {prompt_msg}")
    print("[TEST CALLBACK] Auto-replying NO")
    return False

def test_file_ops():
    print("Testing Phase 4: File Operations Tool\n")
    
    # 1. Setup sandbox
    sandbox = os.path.expanduser("~/echelon_sandbox")
    if os.path.exists(sandbox):
        shutil.rmtree(sandbox)
    os.makedirs(sandbox)
    
    file_path = os.path.join(sandbox, "test.txt")
    dest_path = os.path.join(sandbox, "test_moved.txt")
    
    # 2. Test Write (Create -> ALWAYS_ASK)
    print("--- Test 1: Write (Create) ---")
    handler = FileOpsHandler(ask_callback=auto_ask_callback)
    res = handler.writers.write_file(file_path, "Hello World from ECHELON")
    print(f"Result: {res}")
    assert os.path.exists(file_path), "File was not created!"
    
    # 3. Test Write (Overwrite -> ALWAYS_DENY)
    print("\n--- Test 2: Write (Overwrite) ---")
    res = handler.writers.write_file(file_path, "This should fail")
    print(f"Result: {res}")
    assert "Permission denied" in res, "Overwrite should be blocked by ALWAYS_DENY!"
    
    # 4. Test Read (ALWAYS_ALLOW)
    print("\n--- Test 3: Read ---")
    res = handler.readers.read_file(file_path)
    print(f"Result: {res}")
    assert "Hello World" in res, "Read content mismatch!"
    
    # 5. Test Move (ALWAYS_ASK, user says YES)
    print("\n--- Test 4: Move (Approved) ---")
    res = handler.manager.move_file(file_path, dest_path)
    print(f"Result: {res}")
    assert os.path.exists(dest_path), "File was not moved!"
    assert not os.path.exists(file_path), "Old file still exists!"
    
    # 6. Test Delete (ALWAYS_DENY)
    print("\n--- Test 5: Delete (Denied) ---")
    res = handler.manager.delete_file(dest_path)
    print(f"Result: {res}")
    assert os.path.exists(dest_path), "File should not be deleted!"
    assert "Security Refusal" in res, "Delete should be blocked by ALWAYS_DENY!"
    
    print("\n✅ All File Operations tests passed! Security constraints upheld.")
    
if __name__ == "__main__":
    test_file_ops()
