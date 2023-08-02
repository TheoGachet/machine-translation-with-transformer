# Machine Translation using attention mechanisms and Transformer model

## Overview

This repository contains the implementation of a Portuguese-to-English translation model. Built on TensorFlow, it utilizes the Transformer architecture to translate sentences from Portuguese to English with high accuracy.

## Features

1. **Transformer Architecture**: This model is built using the state-of-the-art Transformer architecture, ensuring high-quality translations.
2. **Attention Visualizations**: Along with translations, the codebase allows users to visualize attention weights, providing insights into how the model processes input sentences.
3. **Export & Reload**: I've included utilities to export the trained model and reload it for future use, making deployment more straightforward.

## Installation & Setup

### Requirements:

- TensorFlow 2.x
- Python 3.7+

### Steps:

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Navigate to the project directory:
```bash
cd path/to/directory
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training:

You can start the training process by running:
```bash
python train.py
```

### Translation & Visualization:

To translate a Portuguese sentence and visualize the attention weights, execute:
```bash
python translate.py --sentence="Your Portuguese sentence here"
```

### Exporting the Model:

Once satisfied with the training results, you can export the model using:
```bash
python export_model.py
```

This will save the model in the `translator/` directory.

### Loading and Using the Saved Model:

To use the saved model for translation, simply run:
```bash
python use_saved_model.py --sentence="Your Portuguese sentence here"
```

## Contribute

I welcome contributions to improve this translator. Whether it's adding new features, improving the model's accuracy, or simply fixing bugs and improving documentation, all contributions are appreciated.

## License

This project is licensed under the MIT License. Refer to the `LICENSE` file for more information.

## Contact

If you have questions, suggestions, or need further assistance, feel free to reach out.

---

I hope you find this Portuguese-to-English translator useful! Your feedback and suggestions will go a long way in making this a robust and reliable tool.
