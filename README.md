# AiFacer

## Overview

This repository contains a AiFacer powered by machine learning models. The detector is designed to analyze videos and determine whether they contain AiFacer content or not. The system utilizes pre-trained models, and the process involves running a Jupyter notebook (`Trainer.ipynb`) to load the necessary models. After that, a server is started to provide a user-friendly interface for detecting content in videos.

## Setup Instructions

Follow these steps to set up and run the AiFacer:

### 1. Open Trainer.ipynb

- Open the Jupyter notebook `Trainer.ipynb` in your preferred environment (e.g., Jupyter Notebook, JupyterLab).

### 2. Run All Cells

- Execute all the cells in the notebook to load the required machine learning models. Note that this step may take a while depending on your system specifications, and having a GPU will significantly speed up the process.

### 3. Start the Server

- Open a terminal or command prompt.
- Run the following command:

    ```bash
    python server.py
    ```

- This will start the server, and you will see output indicating that the server is running.

### 4. Access the Web Interface

- Open your web browser and go to `http://localhost:5000`.
- You should see a user interface for the AiFacer.

### 5. Upload Video and Enter Text

- Use the provided interface to upload a video file (in .mp4 or .mpg format).
- Enter the text that you want to check for AiFacer content.

### 6. Get Results

- Click the "Detect" button.
- The system will analyze the video and text, and it will provide feedback on whether the content is real or fake.

## Note

- For optimal performance, it is recommended to run this on a system with a GPU.

- If you encounter any issues, refer to the troubleshooting section in the documentation or create an issue on the GitHub repository.

Enjoy using the AiFacer! Feel free to contribute to the project and share your feedback.

## License

This project is licensed under the [MIT License](LICENSE).
