Mostafa  [7:22 AM]
2. As you mentioned, WFM uses the arithmetic mean of the three available MRI modalities in wavelet space as the starting point. But why should the arithmetic mean of T1, T1c, T2 and FLAIR (whichever three are available) be the optimal starting point for synthesizing the missing modality? Different sequences contain complementary information. For example, T1c provides information about contrast enhancement, while FLAIR is particularly useful for edema-related abnormalities. A fixed arithmetic average does not adapt to these differences. Instead, we could investigate a target-aware informed source:
$$
s =  \sum(alpha_{m,b}(c,y) c_{m,b}] (for m = 1, 2, 3 modalities and b = low-freq and high-freq subbands)
L = mse ()... + s
$$
where alpha are learned weights conditioned on the available modalities c and the target modality y across wavelet subbands b. This would allow different modalities to contribute differently to the low- and high-frequency information used to synthesize each target.
However, a learned starting point remains a deterministic function of the conditioning modalities. It cannot overcome the conditional-mean limitation we discussed earlier. Its potential advantage is better inductive bias, easier residual prediction or improved optimization, not access to additional information.
