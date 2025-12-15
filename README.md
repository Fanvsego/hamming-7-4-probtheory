# Hamming(7,4) Code Simulation & Information Theory Analysis

This repository contains a Python simulation of the **Hamming(7,4)** error-correcting code. It models the entire pipeline of message encoding, transmission through a noisy channel, and decoding. Additionally, it analyzes the system using fundamental concepts from Information Theory, including Shannon Entropy, Information Quantity, and Transmission Efficiency.

## Project Overview

The project simulates the transmission of 4-bit messages encoded into 7-bit codewords. It introduces random bit errors based on a probability parameter $P$ and attempts to correct single-bit errors using the Hamming algorithm. The simulation generates plots to visualize:
1. Error distribution (histogram).
2. Entropy ratios (Received vs. Source).
3. Information quantity ratios.
4. Transmission efficiency relative to error probability.

## Mathematical Background

### Hamming(7,4) Encoding

The Hamming(7,4) code encodes a **4-bit data message** $\mathbf{m}$ into a **7-bit codeword** $\mathbf{c}$. This is achieved using a Generator Matrix $G$.

Let the message be $\mathbf{m} = (a_1, a_2, a_3, a_4)$. The encoding process is defined as:

$$
\mathbf{c} = \mathbf{m} \cdot G \pmod 2
$$

Where the Generator Matrix $G$ is:

$$
G = \begin{pmatrix}
1 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 1 & 1 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 & 1 & 0 \\
1 & 1 & 0 & 1 & 0 & 0 & 1 \\
\end{pmatrix}
$$

The resulting codeword is $\mathbf{c} = (x_1, x_2, x_3, x_4, x_5, x_6, x_7)$.

### Noisy Channel Model

The encoded message is transmitted through a **Binary Symmetric Channel (BSC)**. Each bit in the codeword has a probability $P$ of being flipped (0 becomes 1, or 1 becomes 0) and a probability $Q = 1 - P$ of remaining correct.

### Syndrome Decoding

Upon receiving the message $\mathbf{r}$ (which may contain errors), we calculate the **Syndrome** vector $\mathbf{z}$ using the Parity Check Matrix $H$.

$$
\mathbf{z} = \mathbf{r} \cdot H^T \pmod 2
$$

Where the transpose of the Parity Check Matrix $H^T$ is:

$$
H^T = \begin{pmatrix}
1 & 0 & 1 & 0 & 1 & 0 & 1 \\
0 & 1 & 1 & 0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1 & 1 & 1 & 1
\end{pmatrix}^T
$$

The position of the error (if any) is determined by converting the binary syndrome vector to a decimal integer:

$$
\text{Position } i = z_1 \cdot 2^0 + z_2 \cdot 2^1 + z_3 \cdot 2^2
$$

If $\mathbf{z} \neq \mathbf{0}$, the bit at position $i$ is inverted to correct the error. The decoded message $\mathbf{m}^*$ is extracted from the corrected codeword.

### Probability Analysis

Let $X$ be the random variable representing the number of corrupted bits.
* **Probability of error per bit:** $P$
* **Probability of no error per bit:** $Q = 1 - P$

The Hamming(7,4) code can correct exactly **1 bit error**. Therefore, a message is decoded correctly if there are **0 errors** or **1 error**.

$$
P(\text{Correct Decode}) = P(X=0) + P(X=1)
$$

Using the Binomial Distribution formula $\binom{n}{k} P^k Q^{n-k}$:

1.  **No errors:** $P(X=0) = Q^7$
2.  **One error:** $P(X=1) = \binom{7}{1} P^1 Q^6 = 7 P Q^6$

The theoretical probability of a correct decoding, denoted here as $P_{\text{correct}}$, is:

$$
P_{\text{correct}} = Q^7 + 7PQ^6 = Q^6(Q + 7P)
$$

### Confidence Interval

*Note: The confidence interval calculation is currently not implemented in the codebase.*

To ensure the simulation results are statistically significant, we calculate the required number of iterations ($N$). We treat the decoding success as a Bernoulli trial.

* **Mean:** $p_{succ}$ (probability of correct decode, derived from $P_{\text{correct}}$ above).
* **Variance:** $\sigma^2 = p_{succ} \cdot (1 - p_{succ})$

Using the Central Limit Theorem (CLT) for a 95% confidence level ($z \approx 1.96$) and a margin of error $\varepsilon = 0.03$:

$$
N = \left\lceil \frac{z^2 \cdot p_{succ} \cdot (1 - p_{succ})}{\varepsilon^2} \right\rceil
$$

## Implementation Notes

There are two different implementations for the Shannon Entropy, Information Quantity, and Transmission Efficiency functions. These implementations depend on the assumptions regarding the message structure and how "useful information" is interpreted.

### 1. Shannon Entropy ($H$)

Entropy measures the uncertainty or "surprise" associated with the message.

#### Scenario 1: Transmitted and received messages are distinct
* **Principle:** This model assumes a scenario where the received message $\mathbf{m}^*$ is never equal to the transmitted message $\mathbf{m}$.
* **Source Probability ($P_m$):** Assumes a uniform distribution for a 4-bit message.
    $$P_m = \left(\frac{1}{2}\right)^4$$
* **Received Probability ($P_{m^*}$):** The probability of receiving a corrupted message (an uncorrectable error occurring).
    $$P_{m^*} = 1 - Q^6(7P + Q)$$
* **Formulas:**
    * $H_{source} = -P_m \log_2(P_m)$
    * $H_{received} = -P_{m^\ast} \log_2(P_{m^\ast})$
    * **Output:** Ratio $\frac{H_{received}}{H_{source}}$

#### Scenario 2: Transmitted and received messages can match
* **Principle:** This model calculates entropy based on the actual outcome of the simulation step. It checks if the decoded message matches the original.
* **Probabilities:**
    * If $\mathbf{m}^* = \mathbf{m}$ (Success): The probability is the theoretical success rate of the Hamming code.
        $$P_{m^*} = Q^6(7P + Q)$$
    * If $\mathbf{m}^* \neq \mathbf{m}$ (Failure): The probability is the theoretical failure rate.
        $$P_{m^*} = 1 - Q^6(7P + Q)$$
* **Formula:** Calculates $H_{received}$ using the specific $P_{m^*}$ determined by the simulation state.

---

### 2. Information Quantity ($I$)

This metric measures the amount of information contained in the message bits.

#### Scenario 1: Transmitted and received messages are distinct
* **Principle:** We assume that the received message is never equal to the transmitted message. This treats the message source not as uniform, but dependent on the previous bit (next bit can be equal with prob of 1/3), and calculates received information based on a global error probability.
* **Source Probability ($P_m$):**
    * Base probability: $0.5$
    * For each subsequent bit:
        * If $bit_i = bit_{i-1}$: multiply $P_m$ by $1/3$.
        * If $bit_i \neq bit_{i-1}$: multiply $P_m$ by $2/3$.
    * $I_{source} = -\frac{1}{4} \log_2(P_m)$ (Normalized by 4 bits).
* **Received Information:**
    $$I_{received} = -\log_2 \left( P \cdot [1 - Q^6(7P + Q)] \right)$$

#### Scenario 2: Transmitted and received messages can match
* **Principle:** We assume that parts of, or the entire, received message can equal the transmitted message. Useful information is calculated by iterating through every bit of the message and checking for correctness individually.
* **Method:**
    1.  Simulate the transmission and decoding.
    2.  Iterate through bits $i = 0$ to $3$.
    3.  If $m[i] = m^*[i]$ (Bit is correct):
        $$I_{received} \mathrel{+}= -\log_2(Q + 7PQ^6)$$
        *(Adds the information value of a correct bit).*
    4.  If $m[i] \neq m^*[i]$ (Bit is wrong):
        $$I_{received} \mathrel{+}= -\log_2(P \cdot [1 - Q^6(7P + Q)])$$
        *(Adds the information value of an error state).*
* **Output:** Ratio $\frac{I_{received}}{I_{source}}$

---

### 3. Transmission Efficiency ($E$)

Efficiency quantifies how much "useful" information passes through the channel relative to the total bandwidth used (7 bits).

#### Scenario 1: Lost information approach
* **Principle:** We define "useful information" here as the information that was not successfully received (lost information).
* **Variables:**
    * $m$: Message length (4).
    * $n$: Codeword length (7).
* **Probabilities:**
    * $P_m = (1/2)^m$
    * $P_r = 1 - Q^{n-1}(Q + n \cdot P)$ (Probability of error for length $n$).
* **Useful Information ($I_{val}$):**
    $$I_{val} = I_{received} - I_{source} = [-\log_2(P_r)] - [-\log_2(P_m)]$$
* **Efficiency:** $\frac{I_{val}}{n}$

#### Scenario 2: Bitwise message intersection
* **Principle:** Calculates efficiency by summing the information only from bits that were actually received correctly. Wrong bits contribute 0 useful information.
* **Method:**
    1.  Perform simulation.
    2.  Initialize $I_{val} = 0$.
    3.  For each bit $i$:
        * If $m[i] = m^*[i]$:
            $$I_{val} \mathrel{+}= -\log_2(Q + nPQ^{n-1})$$
        * Else:
            $$I_{val} \mathrel{+}= 0$$
* **Efficiency:** $\frac{I_{val}}{n}$
* **Zero Efficiency Point:** This implementation is used to find the specific probability $P$ where the efficiency curve crosses zero, indicating the channel has become useless.
