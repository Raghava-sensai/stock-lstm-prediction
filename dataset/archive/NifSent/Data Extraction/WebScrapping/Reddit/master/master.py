import subprocess

# List of full paths to the Python scripts
scripts = [
    r".\Reddit\r-worldnews\Scrapper.py",
    r".\Reddit\r-news\Scrapper.py",
    r".\Reddit\r-indianews\Scrapper.py"
]

for script in scripts:
    try:
        print(f"Running {script}...")
        # Run the script using its full path
        subprocess.run(["python", script], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break  # Stop execution if a script fails
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break
