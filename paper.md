---
title: 'Hybrid Exponential Distributions: A Python Package for Advanced Probability Modeling'
tags:
  - Python
  - statistics
  - probability distributions
  - reliability analysis
  - maximum likelihood estimation
authors:
  - name: Ayeni Taiwo Michael
    orcid: 0000-0002-6823-1417
    corresponding: true
    affiliation: "1"
affiliations:
 - name: Independent Researcher
   index: 1
date: 14 August 2025
bibliography: paper.bib
---

# Summary

The `hybrid-exponential-distributions` package provides Python implementations of four novel hybrid probability distributions involving exponential distributions, developed by Ogunwale et al. [@Ogunwale2019], Ogunwale et al. [@Ogunwale2022], Ayeni et al. [@Ayeni2023], and Adisa et al. [@Adisa2025]. These distributions Exponential-Gamma Distribution (EGD), Exponential-Exponential Distribution (EED), Rayleigh-Exponential-Gamma Distribution (REGD), and Exponential-Gamma-Rayleigh Distribution (EGRD) extend classical probability theory for reliability engineering, survival analysis, and risk assessment.

The package implements robust maximum likelihood estimation, model comparison tools, and specialized visualization capabilities, addressing the need for flexible distributions that model complex failure mechanisms and non-monotonic hazard rates.

# Statement of need

Traditional probability distributions often inadequately model complex real-world phenomena. The exponential distribution assumes constant hazard rates, while gamma distributions may not capture observed tail behaviors in reliability data. Hybrid distributions combine multiple distribution strengths, providing greater flexibility and better empirical fits.

Recent theoretical work by Ogunwale et al. [@Ogunwale2019; @Ogunwale2022], Ayeni et al. [@Ayeni2023], and Adisa et al. [@Adisa2025] developed hybrid exponential distributions, but no comprehensive software implementation existed, hindering practical application in statistical modeling. The package fills this gap by providing efficient implementations, maximum likelihood estimation, AIC/BIC model comparison, goodness-of-fit testing, and reliability analysis tools.

# Implementation

The package follows object-oriented design with `BaseHybridDistribution` providing unified interfaces. The four distributions employ different hybridization methodologies:

**EGD** [@Ogunwale2019]: Product methodology, PDF: $f(x) = \frac{\lambda^{\alpha+1} x^{\alpha-1} e^{-\lambda x}}{\Gamma(\alpha)}$

**EED** [@Ogunwale2022]: ED-X methodology, PDF: $g(x) = \lambda^2 e^{-\lambda^2 x}$

**REGD** [@Ayeni2023]: T-X approach with Rayleigh baseline, PDF: $f(x) = \frac{\lambda^{\alpha+1} x^{\alpha-1} 2^{\alpha} \theta}{\sigma^2 2^{\alpha} \Gamma(\alpha) - \lambda\gamma(\alpha,x)} \exp\left(-\frac{4\lambda x\sigma^2 - \theta^2}{2\sigma^2}\right)$

**EGRD** [@Adisa2025]: T-X methodology with Exponential-Gamma baseline, PDF: $g(x) = \frac{\theta^{\alpha} 2\lambda x^{2\alpha-1}}{\Gamma(\alpha)} \exp(-2\theta x)$

The T-X (Transformer-X) methodology, developed by Alzaatreh et al. [@Alzaatreh2013], generates new distributions using $g(x) = \frac{f(x)}{F(x)} \left(-\log(1-F(x))\right)^{\alpha-1} \exp\left(-\lambda\left(-\log(1-F(x))\right)^{\beta}/2\right)$ where $f(x)$ and $F(x)$ are the PDF and CDF of a baseline distribution.

Core functionality includes L-BFGS-B optimization for parameter estimation, factory pattern for distribution management, Kolmogorov-Smirnov testing, and visualization with PDF, CDF, survival, and hazard rate plots. Vectorized numpy operations ensure computational efficiency.

# Applications

The package addresses needs in reliability engineering (component lifetime modeling), survival analysis (time-to-event data), quality control (statistical process control), risk assessment (financial modeling), and academic research requiring flexible probability models. Applications demonstrate superior fits compared to classical distributions.

# Acknowledgements

I acknowledge the theoretical foundations provided by the original research authors and thank potential reviewers for their valuable feedback.

# References