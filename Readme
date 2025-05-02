# Code Archive for NCOMMS-25-08422-T

This repository contains the Python code for the analysis in "NCOMMS-25-08422-T". It simulates Ultra-Fast Charging Station (UFCS) deployment, optimizes Energy Storage System (ESS) operation, and assesses grid risks.

## File Descriptions:

*   **`UFCSsim.py`**: Simulates upgrading charging stations to UFCS in specific cities (Beijing, Guangzhou, Shanghai). It processes charging records and station data to:
    *   Select guns for upgrade based on type and power increase potential.
    *   Adjust charging times for faster UFCS speeds.
    *   Generate pre- and post-upgrade load profiles for risk/ESS analysis.

*   **`ESSopt.py`**: Optimizes ESS scheduling for charging stations using Google OR-Tools to minimize costs from UFCS. It:
    *   Uses city-specific electricity tariffs (time-of-use, peak demand).
    *   Processes load data from `UFCSsim.py`.
    *   Calculates required ESS parameters (capacity, power).
    *   Solves for the least-cost ESS charge/discharge schedule considering efficiency and constraints.

*   **`Risk.py`**: Quantifies grid over-ramping risks from UFCS load variability using profiles from `UFCSsim.py`. It:
    *   Calculates the load difference (ramping load) between baseline and UFCS scenarios.
    *   Estimates the probability of exceeding a ramp rate threshold using frequency-based and KDE methods.