import gradio as gr



"""
### BLOCK EXAMPLE
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Say Hello")
    greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")
"""

"""
#### Different imput types dropdownm radio, slider. checkbox
def sentence_builder(quantity, animal, countries, place, activity_list, morning):
    return f"The {quantity} {animal}s from {" and ".join(countries)} went to the {place} where they {" and ".join(activity_list)} until the {"morning" if morning else "night"}"


demo = gr.Interface(
    sentence_builder,
    [
        gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
        gr.Dropdown(
            ["cat", "dog", "bird"], label="Animal", info="Will add more animals later!"
        ),
        gr.CheckboxGroup(["USA", "Japan", "Pakistan"], label="Countries", info="Where are they from?"),
        gr.Radio(["park", "zoo", "road"], label="Location", info="Where did they go?"),
        gr.Dropdown(
            ["ran", "swam", "ate", "slept"], value=["swam", "slept"], multiselect=True, label="Activity", info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl."
        ),
        gr.Checkbox(label="Morning", info="Did they do it in the morning?"),
    ],
    "text",
    examples=[
        [2, "cat", ["Japan", "Pakistan"], "park", ["ate", "swam"], True],
        [4, "dog", ["Japan"], "zoo", ["ate", "swam"], False],
        [10, "bird", ["USA", "Pakistan"], "road", ["ran"], False],
        [8, "cat", ["Pakistan"], "zoo", ["ate"], True],
    ]
)
"""


#### FUNCTION : HERE TEXT DIFFERENCE
from difflib import Differ

def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]


demo = gr.Interface(
    diff_texts,
    [
        gr.Textbox(
            label="Text 1",
            info="Initial text",
            lines=3,
            value="The quick brown fox jumped over the lazy dogs.",
        ),
        gr.Textbox(
            label="Text 2",
            info="Text to compare",
            lines=3,
            value="The fast brown fox jumps over lazy dogs.",
        ),
    ],
    gr.HighlightedText(
        label="Diff",
        combine_adjacent=True,
        show_legend=True,
        color_map={"+": "red", "-": "green"}),
    theme=gr.themes.Base()
)

if __name__ == "__main__":
    demo.launch()
