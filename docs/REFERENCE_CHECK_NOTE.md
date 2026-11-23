# Reference check (15 Oct 2026)
Checked against primary records (arXiv abstract pages, Crossref API) unless marked.
| # | Reference | Result |
|---|---|---|
| 1 | Choukse et al., arXiv:2508.14318 | Exists; 57 authors. Cited as "E. Choukse et al." |
| 2 | Latif et al., arXiv:2412.08602 | Title and authors match; peak ~8.4 kW in abstract |
| 3 | Newkirk et al., arXiv:2506.14551 | Title and authors match; full text gives idle 1.86 kW and highest node average 7.79 kW |
| 4 | Radovanovic et al. | CORRECTED. arXiv:2106.11750 has 15 authors (Radovanovic, Koningstein, Schneider, Chen, Duarte, Roy, Xiao, Haridasan, Hung, Care, Talukdar, Mullen, Smith, Cottman, Cirne). Published as IEEE Trans. Power Systems 38(2):1270-1280, 2023, DOI 10.1109/TPWRS.2022.3173250 |
| 5 | Hogade, Pasricha, Siegel | CORRECTED. IEEE Trans. Sustainable Computing 7(2):400-413, 2022, DOI 10.1109/TSUSC.2021.3086087 (three authors: Crossref lists Siegel as third) |
| 6 | Deylamsalehi et al., JOCN 10(10) | Crossref: title, six authors, volume/issue and start page D72 match. End page D83 not shown in the record |
| 7 | Colangelo et al., Nature Energy | Crossref: 16 authors, vol. 11, no. 2, pp. 254-261; online Dec 2025, issue dated 2026 |
| 8 | Won et al., ASTRA-sim2.0, ISPASS 2023 | Crossref: pp. 283-294, DOI 10.1109/ISPASS57527.2023.00035 (page numbers added) |
| 9 | Riley and Henderson, ns-3 | Not re-checked here (supervisor's audit: correct) |
| 10 | Law, Simulation Modeling and Analysis, 5th ed. | Not re-checked here (supervisor's audit: correct) |
| 11a | Li et al., OSDI 2026 | Paper (supervisor's audit: on the OSDI 2026 programme) |
| 11b | Alibaba Cluster Trace GPU v2026 | Dataset, cited separately from the paper (README of the public repository read) |
| 12 | Samsi et al., HPEC 2021 | Crossref lookup was rate-limited; not re-checked here (supervisor's audit: correct) |

## Re-check, 21 Oct 2026
| # | Reference | Result |
|---|---|---|
| 4 | Radovanovic et al. | 15 authors confirmed again (dblp and arXiv:2106.11750). Venue confirmed as IEEE Trans. Power Systems. Vol. 38(2), pp. 1270-1280 not re-confirmed from a primary record today (Crossref lookup failed): **check on IEEE Xplore, DOI 10.1109/TPWRS.2022.3173250, before sign-off.** |
| 5 | Hogade, Pasricha, Siegel | arXiv:2106.00066 lists three authors: Ninad Hogade, Sudeep Pasricha, Howard Jay Siegel. The review names "Hogade and Pasricha"; the three-author form follows the primary record and is kept. |
| 8 | Won et al., ASTRA-sim2.0 | ISPASS 2023, pp. 283-294, DOI 10.1109/ISPASS57527.2023.00035: confirmed by dblp and the authors' citation entry in the ASTRA-sim repository. |
| 12 | Samsi et al., MIT Supercloud | HPEC 2021, pp. 1-8, DOI 10.1109/HPEC49654.2021.9622850: confirmed by the dataset's official repository citation; 27 authors per arXiv:2108.02037. Add the DOI. |

## Full re-check, 24 Oct 2026 (all 13 entries of proposal v5_3)
| # | Reference | Primary record checked | Result |
|---|---|---|---|
| 1 | Choukse et al., arXiv:2508.14318 | arXiv abstract page | Title matches; 57 authors, first E. Choukse; 2025 |
| 2 | Latif et al., arXiv:2412.08602 | arXiv abstract page | Title and all 8 authors match; 2024 |
| 3 | Newkirk et al., arXiv:2506.14551 | arXiv abstract page | Title and all 7 authors match; 2025 |
| 4 | Radovanović et al., IEEE TPWRS | arXiv:2106.11750 (15 authors) and Crossref title search in ISSN 0885-8950 | **Confirmed**: vol. 38, no. 2, pp. 1270-1280, 2023, DOI 10.1109/TPWRS.2022.3173250 (Crossref indexes the DOI in lower case; the direct lookup fails, the title search finds it) |
| 5 | Hogade, Pasricha, Siegel, IEEE TSUSC | Crossref by DOI | Three authors, vol. 7, no. 2, pp. 400-413, 2022: matches |
| 6 | Deylamsalehi et al., JOCN | Crossref by DOI and Optica abstract page | Six authors, vol. 10, no. 10, pp. D72-D83, 2018: matches (end page D83 confirmed by Optica) |
| 7 | Colangelo et al., Nature Energy | Crossref by DOI | 16 authors, vol. 11, no. 2, pp. 254-261; online 5 Dec 2025, issue Feb 2026: matches |
| 8 | Won et al., ASTRA-sim2.0, ISPASS 2023 | Crossref title search | Six authors, pp. 283-294, DOI matches |
| 9 | Riley and Henderson, ns-3 | Crossref title search | Springer 2010, pp. 15-34: matches; DOI 10.1007/978-3-642-12331-3_2 added |
| 10 | Law, Simulation Modeling and Analysis | McGraw-Hill Education product page | 5th ed., 2015, ISBN 978-0-07-340132-4, sole author A. M. Law: matches |
| 11a | Alibaba Cluster Trace GPU v2026 | github.com/alibaba/clusterdata | cluster-trace-gpu-v2026 listed; it accompanies the OSDI '26 paper below |
| 11b | Li et al., OSDI 26 | USENIX presentation page | Title matches; 20 authors, first Suyi Li; pp. 2187-2203; USENIX Association, Seattle; pages added |
| 12 | Samsi et al., HPEC 2021 | arXiv:2108.02037 and Crossref by DOI | All 27 authors match; pp. 1-8; DOI matches |

Formatting fixed in v5_3: acronyms in titles (AI, GPU, MIT, Alibaba) were printed in lower case; now braced so they keep their capitals.
