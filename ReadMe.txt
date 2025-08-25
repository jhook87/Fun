What it does: builds a toy “two black holes” model: two n-qubit registers A and B are scrambled, then a fraction p of qubits are swapped to mimic exchanged Hawking modes. It measures three diagnostics as p is dialed: mutual information I(RA:RB), logarithmic negativity across RA|RB, and a simple OTOC proxy.

What you should see: as p increases, correlations rise and a knee (a “threshold” p*) may appear where recovery becomes plausible in more complete models.

How to run (after installing deps):
python qew_experiment.py
It prints a small table and saves a plot to qew_results.png.

This is a classical simulation of very small Hilbert spaces—intended to make the QEW idea operational and testable on a laptop, not to reproduce gravity.
1. Install

________________


Requires Python 3.10+ (3.11 recommended).
Create a virtual environment (recommended):
   * macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

   * Windows (PowerShell)
py -3 -m venv .venv
.venv\Scripts\Activate.ps1

Install minimal dependencies:
pip install -r requirements.txt
The requirements are intentionally light: numpy, scipy, matplotlib.
      2. Quick start

________________


Run with defaults:
python qew_experiment.py
Outputs:
         * Console table with columns: p, I(RA:RB), Negativity, OTOC.

         * Plot saved as qew_results.png showing the three diagnostics vs p.

If you don’t see a figure file, check that you have write permission in the repo folder.
            3. What the script actually does (high level)

System
               * Two registers, A and B, each with n qubits (default n=4).

               * Initial state is pure, typically |000…0⟩; a “diary” qubit may be optionally injected on A (kept off by default in v0).

               * Each register is scrambled by a sequence of independently sampled Haar-random local unitaries (depth set by scramble_steps).

               * Exchange step: every cycle, k qubits are swapped between A and B (k is the “exchanged modes” count). The exchange fraction is p = k / n. We sweep k to sweep p.


Diagnostics
                  * Mutual information I(RA:RB): standard von Neumann entropy combination I = S(RA) + S(RB) − S(RARB). Computed by exact partial traces for these small systems.

                  * Logarithmic negativity across RA|RB: measures distillable entanglement proxy between the two radiation registers (partial transpose eigenvalue sum in log form). For tiny systems it’s cheap and robust.

                  * OTOC proxy: C(t) ~ 1 − Re(tr(W(t) V W(t) V)) over a tiny subsystem; here implemented as a simple commutator-squared style diagnostic evaluated at a fixed short “time” t set by scramble depth. It’s not a full Keldysh-time OTOC—just a chaos witness that monotonically tracks scrambling in this toy model.

Outputs
                     * A terse printed table for each tested p.

                     * A Matplotlib figure with three curves: I(RA:RB), log-negativity, and OTOC vs p.

Expectation
                        * For small n you will often see a gradual rise rather than a crisp phase transition; with slightly larger n and more scramble depth, a visible knee can appear. That knee is our toy “p*” (connected-wedge onset in spirit).

                           4. Configuration knobs you can change

________________


Open qew_experiment.py and look near the top for the CONFIG section (simple constants). Typical knobs:
                              * n: number of qubits per side (default 4). 6–8 is still OK on a laptop; >10 starts to slow down.

                              * scramble_steps: how many layers of random local unitaries to apply per cycle (default small, e.g., 2–3).

                              * k_values: list of integers to sweep (how many qubits to exchange). p = k/n will be plotted for each k in the list.

                              * samples: number of random repeats per p to average out fluctuations (default small for speed).

                              * seed: random seed for reproducibility (set to an int to get repeatable results).

                              * plot: boolean to control whether to write qew_results.png.

If you prefer command-line flags, you can wrap these in argparse easily; the file is structured so converting the CONFIG section into CLI options is straightforward (but the baseline provided runs without any arguments).
                                 5. Scientific notes and limitations

________________


                                    * This is a toy—no gravity, no backreaction, no quantum extremal surfaces; just a fast-scrambling surrogate (Haar circuits), an exchange operation, and standard entanglement diagnostics.

                                    * The OTOC here is a minimal proxy at a fixed effective time set by scramble depth. Do not interpret its absolute value as a true Lyapunov exponent; use it only as a monotone vs. p in this model.

                                    * Log-negativity is calculated across the RA|RB bipartition for the full state of the two registers (after tracing out any ancillas used inside the routine). It’s a good sanity check that correlations are genuinely quantum, not just classical.

                                    * Mutual information can be nonzero even for classical correlations; that’s why we track negativity as well. (A future version will add reflected entropy S_R as a more geometry-aware witness.)

                                    * Small-n effects: with n=4 you may see flat or noisy curves depending on seed; increase n, scramble_steps, or samples for clearer structure, at the cost of runtime.

                                       6. Interpreting the plot

________________


                                          * I(RA:RB) rising with p indicates growing correlations between the two radiation registers as more modes are exchanged.

                                          * Log-negativity > 0 indicates shared quantum entanglement, i.e., an “open” quantum channel resource.

                                          * The OTOC proxy typically decreases (more scrambling) as you crank depth; in the p-sweep it serves as an internal consistency check that you’re not in a trivially coherent regime.

A qualitative “window-opening” signature is: negligible negativity at small p, then a distinct rise beyond some p*; mutual information usually rises earlier, but negativity provides the stricter (quantum) switch.
                                             7. Reproducibility

________________


                                                * Set seed in the CONFIG block for deterministic runs.

                                                * Keep n, scramble_steps, and k_values identical between runs you want to compare.

                                                * For publication-quality plots, set samples to >= 50 and include error bars (easy to add; see the averaging code block).

                                                   8. Performance tips

________________


                                                      * Exact partial traces scale as O(2^(2n)). Keep n small while you iterate on ideas (n=4..8).

                                                      * Use samples=3..10 for quick feedback; increase later for smooth curves.

                                                      * If you bump n, consider limiting the size of the exchanged subset whose negativity you compute (a future variant provides subsystem-restricted diagnostics).

                                                         9. Roadmap (what’s coming next)

________________


v0.2
                                                            * Add a Petz / twirled-Petz recovery attempt for a one-qubit “diary” injected into A, reporting recovery fidelity on B vs p.

                                                            * Add reflected entropy S_R(RA:RB) via canonical purification for small subsystems.

v0.3
                                                               * Switch OTOC proxy to a proper forward-backward echo on a tagged two-qubit operator with a tunable “time” parameter.

                                                               * Optional noise channels (depolarizing, amplitude damping) to test robustness.

v0.4
                                                                  * Simple SYK-like random-four-fermion surrogate layer as an alternative scrambler.

                                                                  * Batch sweeps and CSV logging for parameter scans.

                                                                     10. How this maps to the QEW idea

________________


                                                                        * The two registers A and B represent two “old” black-hole radiation sectors RA and RB.

                                                                        * Haar random layers mimic fast scrambling (Hayden–Preskill spirit).

                                                                        * The SWAP of k out of n qubits implements “exchanged Hawking modes” between the horizons; p = k/n is your control knob.

                                                                        * The rise of quantum correlations (negativity > 0 alongside MI > 0) as p increases is the toy counterpart of “connected entanglement wedge turns on once exchange exceeds a threshold p*”.

                                                                        * No-signaling: there is no classical side channel here, so nothing in this script can be used to signal; it only diagnoses entanglement resources.

                                                                           11. Typical questions

________________


Q: My curves are flat (near zero).
A: Increase n to 6–8, increase scramble_steps (e.g., 3–5), try more samples, and try a different seed.
Q: The OTOC barely moves.
A: At small sizes it’s just a consistency witness. Treat the MI and negativity as primary in v0.
Q: Can I run 12+12 qubits?
A: It will get slow due to exact density-matrix ops. Start small, profile, and scale cautiously.
Q: Where’s the decoder?
A: Intentional for v0. The file is organized so that adding a Petz/Yoshida–Kitaev recovery block is straightforward—this is slated for v0.2.
                                                                              12. File layout (minimal)

________________


                                                                                 * qew_experiment.py
Contains: state construction, random local unitaries, swapping k qubits, partial trace utilities, von Neumann entropy, mutual information, logarithmic negativity, simple OTOC proxy, the p-sweep runner, and plotting.

                                                                                 * requirements.txt
numpy, scipy, matplotlib

                                                                                 * qew_results.png
Produced on each run unless plotting is disabled.

                                                                                    13. Suggested workflow in your GitHub repo

________________


                                                                                       1. Drop qew_experiment.py, requirements.txt, and this README into the repo (e.g., jhook87/Fun).

                                                                                       2. (Optional) Add a .gitignore with standard Python entries (e.g., pycache/ and .venv/).

                                                                                       3. Commit and push.

                                                                                       4. On your machine: create a venv, pip install -r requirements.txt, run python qew_experiment.py.

                                                                                       5. Inspect qew_results.png, tweak CONFIG, iterate.

                                                                                       6. When ready, open an “Experiments” folder and start saving CSVs/PNGs for different seeds and (n, depth, p) grids.

                                                                                          14. License and citation

________________


                                                                                             * License: choose what you prefer (MIT is a good default).

                                                                                             * If you write this up, please cite your own QEW notes and the usual modern black-hole information references that motivated this toy (Page curve, Hayden–Preskill, islands/QES, reflected entropy, traversable-wormhole teleportation). This code is just a pedagogical bridge from those ideas to a runnable sandbox.

                                                                                                15. Safety/ethics

________________


This repository is a scientific sandbox. It does not enable signaling, faster-than-light communication, or any real-world action beyond numerical experiments. Keep results in the proper theoretical context.
Contact
If you want me to wire in the Petz/Yoshida–Kitaev decoder and reflected entropy S_R next, say the word and I’ll ship v0.2 with the same interface.