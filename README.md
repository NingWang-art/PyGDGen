# PyGDGen - Python Gradient Descent Generator for Atomic Configuration
PyGDGen, standing for Python Gradient Descent Generator, is a Python package that specializes in optimizing the configuration of clusters with diverse shapes using gradient descent techniques. This tool is adept at creating condensed configurations and allows for extensive customization of constraint boxes for various scientific and engineering applications.

## Features
- **Gradient Descent Optimization**: Uses gradient descent methods for precise and efficient optimization of cluster shapes.
- **Versatile Cluster Shape Handling**: Capable of optimizing multiple and customized cluster shapes.
- **Dense Configuration Generation**: Generates highly condensed configurations, surpassing the capabilities of traditional random generation methods.
- **Customizable Constraint Boxes**: Supports user-defined constraint environments, including periodic conditions, neck-like structures, or non-periodic environments.
- **Wide Application Range**: Ideal for first principle calculations, molecular dynamics simulations, and finite element simulation structures.

## Getting Started

### Prerequisites
- Python 3.7 or later
- Dependencies: torch, numpy, scipy

### Installation
Install PyGDGen via pip for easy use:
pip install pygdgen


### Or clone the Repository
For development purposes, the repository can be cloned as follows:

git clone https://github.com/NingWang-art/PyGDGen.git
cd PyGDGen

## Usage
Basic usage instructions for PyGDGen can be found in the Usage section of our package documentation.

## Demo
Generate configuration containing three big and ten small wulff clusters in non-periodic boundary condition:

https://github.com/NingWang-art/PyGDGen/assets/84500213/733fe0cb-2096-4dfd-bbf1-b9f4c828e928

Generate configuration containing one big and fifty small wulff clusters in periodic boundary condition:

https://github.com/NingWang-art/PyGDGen/assets/84500213/00de4bb6-6cc0-4315-9283-34465903741f

Generate neck configuration containing two fixed clusters and eight small clusters around them:

https://github.com/NingWang-art/PyGDGen/assets/84500213/ba274155-287c-4cdb-aeb9-0f7455b89be9

sixteen small clusters around the neck:

https://github.com/NingWang-art/PyGDGen/assets/84500213/5f67d76d-7d03-4a88-b3ec-27bd94145eb2

## Contributing
Contributions are warmly welcomed and highly appreciated. Please refer to our contributing guidelines for details on the code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for more details.

## Acknowledgments
Extend credits to inspirations, code snippets, etc.
Acknowledge collaborators and significant contributors.
Contact
For further inquiries, feel free to contact NingWang - a320873529@gmail.com.
