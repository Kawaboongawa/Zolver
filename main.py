import atexit
import os
import tempfile

import streamlit as st
from streamlit.logger import get_logger

from Puzzle.Puzzle import Puzzle

LOGGER = get_logger(__name__)

def run():
    IMAGE = None
    st.set_page_config(
        page_title="Zolver",
        page_icon="👋",
    )

    st.write("# Welcome to Zolver! 👋")
    green_screen_toggle = st.toggle("Green Screen?")

    uploaded_file = st.file_uploader("Choose a file")
    image_placeholder = st.empty()

    if uploaded_file is not None:
        # To read file as bytes:
        IMAGE = uploaded_file.getvalue()
        image_placeholder.image(IMAGE)

    if st.button("Solve"):
        if IMAGE is None:
            st.warning('Image is None please upload an image first', icon="⚠️")
            return
        with tempfile.NamedTemporaryFile(delete_on_close=False) as fp:
            fp.write(IMAGE)
            fp.close()
            with st.spinner("Wait for it...", show_time=True):
                puzzle = Puzzle(fp.name, green_screen=bool(green_screen_toggle))
                puzzle.solve_puzzle()
                image_placeholder.image(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored.png"))

if __name__ == "__main__":
    # Create and use temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    os.environ["ZOLVER_TEMP_DIR"] = temp_dir.name
    atexit.register(temp_dir.cleanup)
    run()