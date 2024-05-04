import os
import shutil

def manage_directory(path, delete_existing=False):
    """Creates a directory if it doesn't exist, optionally deletes existing content.

  Args:
      path: The path to the directory.
      delete_existing: If True, deletes existing files/subdirectories in the path before creation (default: False).
  """
    if os.path.exists(path):
        if delete_existing:
            print(f"Directory '{path}' already exists. Clearing contents...")
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Use shutil for recursive directory deletion
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
    else:
        print(f"Creating directory '{path}'...")
        os.makedirs(path, exist_ok=True)  # Create directories recursively with exist_ok=True


import random

def randomize_start_end(start, end, duration):
  """
  This function randomly changes the start time within the allowed range while
  maintaining the duration.

  Args:
      start: The original start time.
      end: The original end time.
      duration: The desired duration between start and end.

  Returns:
      A tuple containing the new start and end times.
  """

  # Calculate the maximum allowed change for the start time
  max_change = end - start - duration

  # Ensure max_change is non-negative (avoid negative durations)
  max_change = max(max_change, 0)

  # Randomly choose a change within the allowed range
  change = random.randint(0, max_change)

  # Update start and end times
  new_start = start + change
  new_end = new_start + duration - 1

  return new_start, new_end

if __name__ == '__main__':
    # Example usage
    start = 10  # Example start time
    end = 20    # Example end time
    duration = 5  # Example desired duration

    new_start, new_end = randomize_start_end(start, end, duration)

    print(f"Original: start = {start}, end = {end}")
    print(f"New: start = {new_start}, end = {new_end}")
