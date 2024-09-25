import argparse
import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd
from datasets import load_dataset


class Lab6Gradio:
    """Lab 6 Gradio class."""

    def __init__(
        self,
        predictions_path: Path = Path("predictions.json"),
        ratings_path: Path = Path("ratings.json"),
        local_path: Path = Path("local"),
        local_file: Path = Path("users.json"),
    ) -> None:
        self.dataset = load_dataset("RekaAI/VibeEval", trust_remote_code=True)
        with open(predictions_path) as f:
            self.predictions = json.load(f)

        with open(ratings_path) as f:
            self.ratings = json.load(f)

        self.local_path = local_path
        if not self.local_path.exists():
            self.local_path.mkdir(parents=True)

        self.local_file = Path(self.local_path, local_file)
        with open(self.local_file, "w") as f:
            json.dump({}, f)

        self.indices = [9, 13, 24, 25, 39, 64, -2, -61, 104, 106]
        self.dataset_size = len(self.predictions)

    def prepare_example_from_index(
        self, username: str, index: int
    ) -> tuple[str, pd.DataFrame, str, str, str, None]:
        """Prepare the example for the given index."""
        example = self.dataset["test"][self.indices[index]]
        input_image = example["image"]

        user_ratings = [val for _, val in self.get_ratings_for_username(username).items()]
        model_ratings = self.ratings[: index + 1]
        ratings = pd.DataFrame(
            columns=["User Ratings", "Model Ratings"],
            data=list(zip(user_ratings, model_ratings, strict=False)),
        )

        prompt_text = example["prompt"]
        groundtruth_text = example["reference"]

        predicted_text = self.predictions[index]
        return input_image, ratings, prompt_text, groundtruth_text, predicted_text, None

    def get_next_example_for_username(
        self, username: str
    ) -> tuple[str, pd.DataFrame, str, str, str, None]:
        """Get the example for the given index."""
        example_index = self.get_example_index_for_username(username)
        new_example_index = min(example_index + 1, self.dataset_size - 1)
        example_tuple = self.prepare_example_from_index(username, new_example_index)
        self.update_example_index_for_username(username, new_example_index)
        return example_tuple

    def on_load(self, request: gr.Request) -> tuple[str, pd.DataFrame, str, str, str, None]:
        """Get the first example for the given username."""
        username = self.get_username_from_request(request)
        current_index = self.get_example_index_for_username(username)
        return self.prepare_example_from_index(username, current_index)

    def on_submit(
        self, request: gr.Request, user_rating: int
    ) -> tuple[str, pd.DataFrame, str, str, str, None]:
        """Update the ratings and get the next example for the given username."""
        username = self.get_username_from_request(request)
        self.update_ratings_for_username(username, user_rating)
        return self.get_next_example_for_username(username)

    def update_ratings_for_username(self, username: str, rating: int) -> None:
        """Update the ratings for the given username."""
        current_index = self.get_example_index_for_username(username)
        user_ratings = self.get_ratings_for_username(username)
        user_ratings[current_index] = rating

        user_file = self.get_user_file(username)
        with open(user_file, "w") as f:
            json.dump(user_ratings, f)

    def get_ratings_for_username(self, username: str) -> dict[int, int]:
        """Get the ratings for the given username."""
        user_file = self.get_user_file(username)
        if not user_file.exists():
            with open(user_file, "w") as f:
                json.dump({}, f)
            return {}

        with open(user_file) as f:
            user_ratings = json.load(f)
        return user_ratings

    def get_example_index_for_username(self, username: str) -> int:
        """Get the example index for the given username."""
        with open(self.local_file) as f:
            users = json.load(f)

        return users.get(username, 0)

    def update_example_index_for_username(self, username: str, example_index: int) -> None:
        """Update the example index for the given username."""
        with open(self.local_file) as f:
            users = json.load(f)

        users[username] = example_index
        with open(self.local_file, "w") as f:
            json.dump(users, f)

    def get_user_file(self, username: str) -> Path:
        """Get the user file for the given username."""
        return Path(self.local_path, f"{username}.json")

    def get_username_from_request(self, request: gr.Request) -> str:
        """Get the username from the request."""
        return request.query_params["USERNAME"]


def main(args: argparse.Namespace) -> None:
    os.environ["GRADIO_TEMP_DIR"] = "gradio_temp_dir"
    app = Lab6Gradio(
        predictions_path=args.predictions_path,
        ratings_path=args.ratings_path,
        local_path=args.local_path,
        local_file=args.local_file,
    )
    theme = gr.themes.Soft(
        spacing_size=gr.themes.sizes.spacing_md, font=gr.themes.GoogleFont("Source Sans Pro")
    ).set(
        button_secondary_background_fill="*neutral_200",
        button_secondary_background_fill_hover="*neutral_300",
    )

    js_lightmode_func = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'light') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
    """

    layout_css = """
    #warning {
        background: #fa5f55;
        color: white;
    }

    .feedback {
        font-size: 16px !important;
    }

    .feedback textarea {
        font-size: 16px !important;
    }

    .app {
        max-width: 1024px !important;
    }

    .markdown {
        max-width: 1024px;
    }

    .markdown td, .markdown th {
        border-color: gray !important;
    }

    .custom-image-container {
        padding-top: 40px; /* Adjust the value as needed */
    }
    """
    with gr.Blocks(theme=theme, js=js_lightmode_func, css=layout_css) as block:
        with gr.Row():
            input_image = gr.Image(
                label="Input Image",
                show_download_button=False,
                container=True,
                interactive=False,
                visible=True,
                elem_classes="custom-image-container",
            )

            ratings = gr.DataFrame(
                headers=["User Ratings", "Model Ratings"],
                col_count=(2, "fixed"),
                interactive=False,
            )

        with gr.Row():
            prompt_text = gr.Textbox(
                label="Prompt",
                interactive=False,
                visible=True,
            )

        with gr.Row():
            with gr.Column():
                groundtruth_text = gr.Textbox(
                    label="Ground Truth Response",
                    interactive=False,
                    visible=True,
                    autoscroll=False,
                )
            with gr.Column():
                predicted_text = gr.Textbox(
                    label="Predicted Response",
                    interactive=False,
                    visible=True,
                    autoscroll=False,
                )

        with gr.Row():
            radio_button = gr.Radio(
                label="Your Rating",
                choices=["1", "2", "3", "4", "5"],
                interactive=True,
                visible=True,
            )

        with gr.Row():
            submit_button = gr.Button(
                value="Submit",
                variant="primary",
                interactive=True,
                visible=True,
            )

        submit_button.click(
            app.on_submit,
            inputs=[radio_button],
            outputs=[
                input_image,
                ratings,
                prompt_text,
                groundtruth_text,
                predicted_text,
                radio_button,
            ],
        )
        block.load(
            app.on_load,
            inputs=[],
            outputs=[
                input_image,
                ratings,
                prompt_text,
                groundtruth_text,
                predicted_text,
            ],
        )
    block.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=Path, default=Path("./data/lab6_predictions.json"))
    parser.add_argument("--ratings_path", type=Path, default=Path("./data/lab6_ratings.json"))
    parser.add_argument("--local_path", type=Path, default=Path("./data/lab6"))
    parser.add_argument("--local_file", type=Path, default=Path("users.json"))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    main(args)
