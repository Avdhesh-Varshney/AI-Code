# 📝 How to Contribute?

Welcome to the **AI-Code** project! Whether you're a seasoned developer or just starting, this guide will help you contribute systematically and effectively. Let's build amazing AI projects together! 🚀

---

## Getting Started

### 🌟 Star This Repository

Show your support by starring the project! 🌟 This helps others discover and contribute. Click [here](https://github.com/Avdhesh-Varshney/AI-Code) to star.

### 🍴 Fork the Repository

Create a personal copy of the repository by clicking the **Fork** button at the top right corner of the GitHub page.

### 📥 Clone Your Forked Repository

Clone your forked repository to your local machine using:

```bash
git clone https://github.com/<your-github-username>/AI-Code.git
```

### 📂 Navigate to the Project Directory

Move into the directory where you've cloned the project:

```bash
cd AI-Code
```

### 🌱 Create a New Branch

Create a separate branch for your changes to keep the `main` branch clean:

```bash
git checkout -b <your_branch_name>
```

---

### 🛠️ Set Up the Development Environment

#### 1. Create a Virtual Environment

To isolate dependencies, create a virtual environment:

```bash
python -m venv myenv
```

#### 2. Activate the Virtual Environment

- **Windows:**
  ```bash
  myenv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source myenv/bin/activate
  ```

#### 3. Install Required Dependencies

Install all dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### 4. Preview Locally 

Use MkDocs to start the development server and preview the project:

```bash
mkdocs serve
```

Access the site locally at:

```
http://127.0.0.1:8000/AI-Code/
```

---

## Making Contributions

### ✏️ Make Changes

Make your desired code edits, add features, or improve documentation. Follow the project's coding standards and contribution guidelines for consistency.

### 💾 Stage and Commit Changes

#### 1. Stage All Changes:

```bash
git add .
```

#### 2. Commit Changes with a Descriptive Message:

```bash
git commit -m "<your_commit_message>"
```

### 🚀 Push Your Changes

Push your branch to your forked repository:

```bash
git push -u origin <your_branch_name>
```

### 📝 Create a Pull Request

1. Navigate to your forked repository on GitHub.
2. Click Pull Requests, then New Pull Request.
3. Select your branch and describe your changes clearly before submitting.

---

## Contribution Guidelines

### 📂 File Naming Conventions

- Use `kebab-case` for file names (e.g., `ai-code-example`).

### 📚 Documentation Standards

- Follow the [PROJECT README TEMPLATE](./project-readme-template.md) and [ALGORITHM README TEMPLATE](./algorithm-readme-template.md).
- Use raw URLs for images and videos rather than direct uploads.

### 💻 Commit Best Practices

- Keep commits concise and descriptive.
- Group related changes into a single commit.

### 🔀 Pull Request Guidelines

- Do not commit directly to the `main` branch.
- Use the PR Template and provide all requested details.
- Include screenshots, video demonstrations, or work samples for UI/UX changes.

### 🧑‍💻 Code Quality Standards

- Write clean, maintainable, and well-commented code.
- Ensure originality and adherence to project standards.

---

## 📘 Learning Resources

### 🧑‍💻 Git & GitHub Basics

- [Forking a Repository](https://help.github.com/en/github/getting-started-with-github/fork-a-repo)
- [Cloning a Repository](https://help.github.com/en/desktop/contributing-to-projects/creating-an-issue-or-pull-request)
- [Creating a Pull Request](https://opensource.com/article/19/7/create-pull-request-github)
- [GitHub Learning Lab](https://lab.github.com/githubtraining/introduction-to-github)

### 💻 General Programming

- [Learn Python](https://www.learnpython.org/)
- [MkDocs Documentation](https://www.mkdocs.org/)
