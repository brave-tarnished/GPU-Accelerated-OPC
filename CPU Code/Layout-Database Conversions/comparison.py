import subprocess
import time
import os
import shutil
import glob

def prepare_test_environment():
    """Prepare test environment with sample DB files"""
    print("Preparing test environment...")
    
    # Create directories if they don't exist
    os.makedirs("DB Files", exist_ok=True)
    os.makedirs("DB Files/Divided", exist_ok=True)
    
    # Check if we need to create sample DB files
    if not glob.glob("DB Files/Divided/*.db"):
        print("Creating sample DB files for testing...")
        # This would normally create test DB files
        # For this example, we'll just copy the same file multiple times
        if os.path.exists("DB Files/multiple_polygon.db"):
            for i in range(4):
                for j in range(4):
                    shutil.copy(
                        "DB Files/multiple_polygon.db", 
                        f"DB Files/Divided/small_layout_block_{i}_{j}.db"
                    )
            print("Created 16 sample DB files")
        else:
            print("Warning: No source DB file found. Tests may fail.")
    else:
        print("Using existing DB files")

def compile_code(source_file, output_name):
    """Compile C++ code with appropriate flags"""
    print(f"Compiling {source_file} to {output_name}...")
    
    # Compile with sqlite3 and pthread libraries
    result = subprocess.run(
        ['g++', source_file, '-o', output_name, '-std=c++17', '-lsqlite3', '-pthread'],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print(f"Compilation Error: {result.stderr}")
        return False
    
    print(f"Successfully compiled {output_name}")
    return True

def run_single_threaded():
    """Run and time the single-threaded version"""
    print("\nRunning single-threaded version...")
    start_time = time.time()
    
    result = subprocess.run(['./single_thread'], capture_output=True, text=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if result.returncode != 0:
        print(f"Runtime Error: {result.stderr}")
        return None
    
    print(f"Single-threaded output: {result.stdout}")
    return elapsed

def run_multi_threaded():
    """Run and time the multi-threaded version"""
    print("\nRunning multi-threaded version...")
    start_time = time.time()
    
    result = subprocess.run(['./multi_thread'], capture_output=True, text=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if result.returncode != 0:
        print(f"Runtime Error: {result.stderr}")
        return None
    
    print(f"Multi-threaded output: {result.stdout}")
    return elapsed

def backup_db_files():
    """Create a backup of DB files before processing"""
    backup_dir = "DB Files/Backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    for db_file in glob.glob("DB Files/Divided/*.db"):
        shutil.copy(db_file, os.path.join(backup_dir, os.path.basename(db_file)))
    
    print("Created backup of DB files")

def restore_db_files():
    """Restore DB files from backup"""
    backup_dir = "DB Files/Backup"
    if os.path.exists(backup_dir):
        for backup_file in glob.glob(f"{backup_dir}/*.db"):
            filename = os.path.basename(backup_file)
            shutil.copy(backup_file, os.path.join("DB Files/Divided", filename))
        
        print("Restored DB files from backup")

def main():
    """Main benchmark function"""
    print("=== Polygon Processing Performance Benchmark ===")
    
    # Prepare test environment
    prepare_test_environment()
    
    # Create backup of original DB files
    backup_db_files()
    
    # Compile both versions
    single_compiled = compile_code("experimental.cpp", "single_thread")
    multi_compiled = compile_code("cpu_multithreading.cpp", "multi_thread")
    
    if not single_compiled or not multi_compiled:
        print("Compilation failed. Cannot proceed with benchmark.")
        return
    
    # Run benchmarks
    single_time = run_single_threaded()
    
    # Restore DB files to original state before running multi-threaded version
    restore_db_files()
    
    multi_time = run_multi_threaded()
    
    # Print results
    if single_time is not None and multi_time is not None:
        print("\n=== Benchmark Results ===")
        print(f"Single-threaded execution time: {single_time:.4f} seconds")
        print(f"Multi-threaded execution time:  {multi_time:.4f} seconds")
        
        if single_time > 0:
            speedup = single_time / multi_time
            print(f"Speedup factor: {speedup:.2f}x")
            
            # Calculate theoretical vs. actual efficiency
            thread_count = len(glob.glob("DB Files/Divided/*.db"))
            if thread_count > 0:
                theoretical_max = min(thread_count, os.cpu_count() or 1)
                efficiency = speedup / theoretical_max * 100
                print(f"Parallel efficiency: {efficiency:.2f}% (compared to theoretical maximum of {theoretical_max}x)")
    else:
        print("Benchmark failed due to runtime errors.")

if __name__ == "__main__":
    main()