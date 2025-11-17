import glob2
import os

from tqdm import tqdm  # Progress bar for iterations
from loguru import logger  # Enhanced logging capabilities

# Remove default logger
logger.remove()
# Add custom logger with colored output
logger.add(
    lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # Timestamp in green
        "<blue>{file}:{line}</blue> | "  # File and line in blue
        "<yellow>{message}</yellow>"  # Message in yellow
    ),
    colorize=True,
)

if __name__ == "__main__":
    sh_file = glob2.glob("generated_uchoa_problem/scripts_gen/*.sh")
    logger.info(f"Found {len(sh_file)} script files to process.")

    for file in sh_file:
        logger.info(f"Processing file: {file}")
        try:
            number = int(file.split("genXML")[1].split(".sh")[0])
            dir_path = f"generated_uchoa_problem/bdd_generated/gen{number}"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"Created directory: {dir_path}")
            else:
                logger.warning(f"Directory {dir_path} already exists, BREAK")
                break

            os.system(f"cp {file} {dir_path}/")
            logger.info(f"Copied {file} to {dir_path}/")
            os.system(f"cp generated_uchoa_problem/generator.py {dir_path}/")
            logger.info(f"Copied generator.py to {dir_path}/")
            os.system(f"bash {dir_path}/{os.path.basename(file)}")
            logger.info(f"Executed script: {dir_path}/{os.path.basename(file)}")
            os.remove(dir_path + "/generator.py")
            logger.info(f"Removed generator.py from {dir_path}/")
            os.remove(dir_path + f"/{os.path.basename(file)}")
            logger.info(f"Removed {os.path.basename(file)} from {dir_path}/")

            vrp_file = glob2.glob("*.vrp")
            logger.info(f"Found {len(vrp_file)} VRP files to move.")
            for vrp in tqdm(vrp_file, desc="Moving VRP files"):
                os.system(f"mv {vrp} {dir_path}/")
        except Exception as e:
            logger.error(f"An error occurred while processing {file}: {e}")
