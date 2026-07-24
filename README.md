# PKU Introduction to Artificial Intelligence — Lab Portfolio (2026)

这是我在北京大学《人工智能引论》课程实验基础上整理的个人作品集。仓库不直接发布课程答案，而是将实验中练习的核心算法重新实现为可独立运行、可测试的通用模块。

## What this repository demonstrates

| Module | Topics | What I implemented |
|---|---|---|
| Search & game playing | DFS, BFS, UCS, A*, adversarial search | Generic graph-search interfaces, path reconstruction, cost-aware search |
| Machine learning from scratch | Logistic regression, numerical stability, gradient descent | NumPy binary classifier with L2 regularization and stable sigmoid |
| NLP & LLM agents | Naive Bayes, TF–IDF retrieval, prompt-based control | Text classification, retrieval ranking and safe API-integration design |
| Robotics | Particle filtering, PD control, RRT planning | Resampling, feedback control and collision-aware sampling-based planning |

## Repository structure

```text
.
├── Lab1-Search-and-Game-Playing/
├── Lab2-Machine-Learning-from-Scratch/
├── Lab3-NLP-and-LLM-Agent/
├── Lab4-Robotics-and-Simulation/
├── docs/                              # Detailed learning notes
├── examples/                          # One-command demonstration
├── src/ai_labs/                       # Independent implementations
└── tests/                             # Deterministic unit tests
```

## Quick start

Requirements: Python 3.10+ and NumPy.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/run_showcase.py
```

## Selected technical takeaways

- Search algorithms are unified by changing only the frontier discipline and priority function.
- Stable numerical implementations matter: clipping or branch-wise sigmoid evaluation avoids overflow.
- TF–IDF and Naive Bayes remain useful transparent baselines for understanding modern NLP systems.
- Robotics algorithms must model uncertainty and physical constraints, not only optimize an abstract objective.
- Reproducibility, tests and scope disclosure are part of an engineering result—not afterthoughts.

## Academic-integrity and attribution notice

The original coursework used teaching frameworks including the UC Berkeley Pacman AI projects. Their license permits educational use but explicitly prohibits publishing solutions. Therefore this repository:

- does **not** contain official handouts, autograders, datasets or solution files;
- does **not** contain student identifiers, grades, model weights or API credentials;
- contains independent implementations with APIs different from the coursework;
- is intended as a portfolio and learning record, not as an answer bank.

Conceptual attribution:

- [UC Berkeley Pacman Projects](https://ai.berkeley.edu/project_overview.html)
- Peking University, *Introduction to Artificial Intelligence* course, Spring 2026

## License

The independent code in this repository is released under the MIT License. Course materials and third-party projects remain under their respective terms.
