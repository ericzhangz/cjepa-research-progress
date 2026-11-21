# Numerical timing run only

This interrupted run used the same objective, initialization, sample RNG and
frozen 2048-update budget as Phase2. It is not the final scientific run.

The tall CUDA SVD dispatched a separate solver per 192-by-4 matrix. A reduced
QR followed by the exact 4-by-4 SVD produces the same G1 densities and first
derivatives without a rank threshold. Analytic node-gradient contraction also
stays in the four response coordinates until after quadrature summation.
Value/gradient and independent finite-difference checks established numerical
equivalence before restarting. Concurrent timing showed a large kernel speedup.

The parent authorized interruption and a complete restart from the original
unlabelled initial offsets and the same sample RNG. This decision used numerical
equivalence and elapsed time; no training-after independent evaluation was read.
The final scientific checkpoint must be the new `A/final.pt` at update 2048.
Full-library monitoring in the new run is limited to initial, 1024 and final;
every update still saves each loss and total/per-prototype gradient statistics.
