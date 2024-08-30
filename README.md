[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url1]

<!-- PROJECT LOGO -->sl_predictors_xai
<br />
<div align="center">
  <h2 align="center">Predictors of Sick Leave: Exploring Hierarchical Determinants and Explainable AI</h2>

  <p align="center">
    Dilan Croos<br>
    <br />
    <a href="https://github.com/dilancroos/sl_predictors_xai"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/dilancroos/sl_predictors_xai/issues">Report Bug</a>
    ·
    <a href="https://github.com/dilancroos/sl_predictors_xai/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project aimed to enhance the interpretability of machine learning (ML) models predicting sick leave (SL) by applying eXplainable Artificial Intelligence (XAI) techniques. The objective was to make complex Artificial Intelligence (AI) models more accessible to non-technical decision-makers, allowing them to understand and act on the data effectively using XAI techniques. I used XAI techniques like Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) to visualize predictions and identify key features influencing SL outcomes. This included the application of these tools to a large dataset of employee sick leave records. My findings demonstrated that XAI significantly improved the clarity of model predictions, providing actionable insights for data scientists and stakeholders. These results will support better decision-making and strategic planning in organizational settings.

In the context of workforce management, predicting and understanding patterns of sick leave (SL) is crucial for improving employee well-being and organizational efficiency. This study identifies the challenge of interpreting complex machine learning (ML) models that predict SL, often seen as “black boxes” by non-technical stakeholders such as HR managers, decision-makers, and employees. These stakeholders need clear, actionable insights to address issues like workplace conditions that contribute to SL, without unintentionally reinforcing biases related to age, gender, or health status.

##### Research Question:

How can eXplainable Artificial Intelligence (XAI) techniques enhance the interpretability of ML models predicting sick leave, to ensure that the insights are transparent, actionable, and ethically sound for non-technical stakeholders?

##### Hypotheses:

1. XAI tools like Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP) can improve stakeholders’ understanding of ML predictions by highlighting key factors influencing SL, making the models more transparent and actionable.

2. Integrating XAI with robust data collection methods will help in identifying and addressing potential discrimination in SL predictions, ensuring the ethical use of AI in workforce management.

All the code and data used in this study are publicly available in a GitHub repository, ensuring transparency and reproducibility. The repository can be accessed at https://is.gd/LtBQcN.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Python 3.11.6^

- install pip

  ```sh
  $ python3 -m pip install pip
  ```

### Installation

1. Clone the repo

   ```sh
   $ git clone git@github.com:dilancroos/sl_predictors_xai.git
   ```

2. Change to the working directory

   ```sh
   $ cd sl_predictors_xai
   ```

- Check <a href="#usage">Usage</a> to create a virtual environment

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

- Create a virtual environment .venv

  ```sh
   $ python -m venv .venv
  ```

- Enter the virtual environment .venv

  ```sh
   $ source .venv/bin/activate
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Dilan Croos - antondilan.crooswarnakulasuriya@cri-paris.org.com

Project Link: [https://github.com/dilancroos/sl_predictors_xai](https://github.com/dilancroos/sl_predictors_xai)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- []()
- []()
- []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/dilancroos/sl_predictors_xai.svg?style=for-the-badge
[contributors-url]: https://github.com/dilancroos/sl_predictors_xai/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dilancroos/sl_predictors_xai.svg?style=for-the-badge
[forks-url]: https://github.com/dilancroos/sl_predictors_xai/network/members
[stars-shield]: https://img.shields.io/github/stars/dilancroos/sl_predictors_xai.svg?style=for-the-badge
[stars-url]: https://github.com/dilancroos/sl_predictors_xai/stargazers
[issues-shield]: https://img.shields.io/github/issues/dilancroos/sl_predictors_xai.svg?style=for-the-badge
[issues-url]: https://github.com/dilancroos/sl_predictors_xai/issues
[license-shield]: https://img.shields.io/github/license/dilancroos/sl_predictors_xai.svg?style=for-the-badge
[license-url]: https://github.com/dilancroos/sl_predictors_xai/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url1]: https://linkedin.com/in/antondilancrooswarnakulasuriya
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com

