import os
import cv2
import glob
import inspect

def get_unique_run_folder(base_folder=None, name=None):
    """Create a unique folder for saving training results based on the script name or a provided name.
    Args:
        base_folder: Base folder where the unique folder will be created (str). 
        name: Optional name for the folder (str). If None, the script name is used.
    Returns:
        unique_folder: Path to the unique folder created (str).
    """
    if name is not None:
        script_name = name
    else:
        try:
            frame = inspect.stack()[1]
            caller_filename = frame.filename
            script_name = os.path.splitext(os.path.basename(caller_filename))[0]
            if script_name in ["ipykernel_launcher", "<ipython-input-", ""]:
                script_name = "notebook_run"
        except Exception:
            script_name = "notebook_run"
    folder = os.path.join(base_folder, script_name)
    suffix = 1
    unique_folder = folder
    while os.path.exists(unique_folder):
        unique_folder = f"{folder}_{suffix}"
        suffix += 1
    os.makedirs(unique_folder, exist_ok=True)
    return unique_folder

def save_figure(fig, epoch, prefix="figure", folder=None):
    """
    Save a matplotlib figure to the specified folder with a unique name based on the epoch and prefix.
    Args:
        fig: matplotlib figure object
        epoch: current epoch number (int)
        prefix: prefix for the filename (str)
        folder: folder to save images (str). If folder is None, a unique run folder is created.
    """
    if folder is None:
        raise ValueError("You must provide a folder. Create it once with get_unique_run_folder() at the start of your script.")
    os.makedirs(folder, exist_ok=True)
    filename = f"{prefix}_{epoch:06d}.png"
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches='tight')
    #print(f"Saved figure: {filepath}")


def make_video(input_folder, output_folder, output="training_montage.mp4", fps=2, prefix=None):
    """
    Combine all PNG images in the folder (optionally with a given prefix) into a video.
    Args:
        input_folder: Folder containing images. If folder is None, a unique run folder is created.
        output_folder: Folder where the video will be saved. If None, it defaults to the input folder.
        output: Output video filename (will be saved in the same folder).
        fps: Frames per second for the video.
        prefix: Only include images whose filenames start with this prefix (str or None).
    """
    if input_folder is None:
        raise ValueError("You must provide an input folder.")

    if prefix is not None:
        pattern = os.path.join(input_folder, f"{prefix}_*.png")
    else:
        pattern = os.path.join(input_folder, "*.png")
    images = sorted(glob.glob(pattern))
    #print(f"Found {len(images)} images for pattern: {pattern}")
    if not images:
        print("No images found in folder:", input_folder)
        return

    # Get frame size from the first image
    frame = cv2.imread(images[0])
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_folder, output)
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    written = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is not None:
            if img.shape[:2] != (height, width):
                #print(f"Resizing {img_path} from {img.shape[:2]} to {(height, width)}")
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            video.write(img)
            written += 1
        else:
            print(f"Warning: Could not read {img_path}")
    #print(f"Total images written to video: {written}")

    video.release()
    print(f"Video saved as {video_path}")

