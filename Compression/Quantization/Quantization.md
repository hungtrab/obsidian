# Phân tích Kỹ thuật Bài báo "Quantization" (R. M. Gray và D. L. Neuhoff)

Dưới đây là bản tóm tắt kỹ thuật chi tiết, tập trung vào các khía cạnh cốt lõi của lý thuyết Lượng tử hóa (Quantization).
Được chia làm 2: [[Quantization-Aware Training]]  (Quantization-Aware Training) và

---

### 1. Bài toán cần giải quyết

Bài báo này giải quyết bài toán cốt lõi là **tối ưu hóa sự đánh đổi giữa tốc độ bit ($R$ - rate) và độ méo tín hiệu ($D$ - distortion)** trong việc ánh xạ tín hiệu liên tục hoặc dữ liệu vector nhiều chiều thành một tập hợp hữu hạn các giá trị rời rạc (lượng tử hóa).

---

### 2. Đóng góp cốt lõi

Đóng góp mới lạ và quan trọng nhất là **hệ thống hóa toàn diện (systematization)** lĩnh vực Lượng tử hóa, cung cấp cái nhìn so sánh và bổ sung lẫn nhau giữa các trụ cột lý thuyết chính:

* **Lý thuyết Lượng tử hóa Tiệm cận/Độ phân giải cao (High-Resolution Theory)**: Cung cấp các công thức tiệm cận (ví dụ: $D \sim 2^{-2R}$) để phân tích hiệu suất bộ lượng tử hóa cố định chiều ($k$ cố định, tốc độ $R \rightarrow \infty$).

* **Lý thuyết Độ méo-Tốc độ Shannon (Shannon Rate-Distortion Theory)**: Thiết lập giới hạn hiệu suất tuyệt đối khi **chiều tăng vô hạn ($k \rightarrow \infty$), tốc độ cố định ($R$)**.

* **Định lý Lloyd/Max** và **Thuật toán Lloyd tổng quát (GLA)**: Cung cấp các điều kiện tối ưu cần thiết (centroid và nearest-neighbor) và quy trình thiết kế thực tế cho bộ lượng tử hóa.

---

### 3. Quy trình của phương pháp

Bài báo mô tả cấu trúc chung của một bộ lượng tử hóa ($q$), đặc biệt là trong mô hình **Vector Quantization (VQ)**, bao gồm ba pha:

* **Bước 1: Mã hóa Tổn hao ($\alpha$ - Lossy Encoding)**
    * **Mục đích:** Chia không gian đầu vào $x \in \mathbb{R}^k$ thành **phân vùng** $\mathcal{S} = \{S_i\}$.
    * **Quy tắc:** $x$ được ánh xạ tới chỉ mục $i$ sao cho $x \in S_i$.
* **Bước 2: Giải mã Tái tạo ($\beta$ - Reproduction Decoding)**
    * **Mục đích:** Gán chỉ mục $i$ cho một **vector tái tạo (codevector)** $y_i$ từ **codebook** $\mathcal{C} = \{y_i\}$.
    * **Quy tắc:** $\hat{x} = \beta(i) = y_i$.
* **Bước 3: Mã hóa Không tổn hao ($\gamma$ - Lossless Encoding)**
    * **Mục đích:** Mã hóa chỉ mục $i$ thành codeword nhị phân $\gamma(i)$ với độ dài $l(\gamma(i))$.
    * **Lượng tử hóa Tốc độ Biến thiên (Variable-Rate):** Sử dụng mã hóa Entropy (ví dụ: Huffman) để tối ưu hóa độ dài codeword dựa trên xác suất $P_i = \text{Pr}(X \in S_i)$.

---

### 4. Hàm mục tiêu và Hàm mất mát

Hàm mục tiêu tổng quát (Operational Lagrangian) được sử dụng để tối ưu hóa sự đánh đổi giữa Độ méo và Tốc độ là:

$$
L(\lambda) = \inf_{(\alpha, \gamma, \beta)} D(\alpha, \beta) + \lambda R(\alpha, \gamma)
$$

#### Giải thích các thành phần:

* $D(\alpha, \beta)$ **(Độ méo trung bình chuẩn hóa):**
    $$
    D(\alpha, \beta) = \frac{1}{k} E[d(X, q(X))] = \frac{1}{k} \sum_{i} \int_{S_i} d(x, y_i) f(x) dx
    $$
    * $d(x, y_i)$: **Hàm Độ méo** (Distortion Measure), thường là **Mean Squared Error (MSE)** $d(x, \hat{x}) = ||x - \hat{x}||^2$ hoặc **Input-Weighted Quadratic Distortion**.
    * Mục đích là tối thiểu hóa sai số trung bình giữa đầu vào $X$ và tái tạo $q(X)$.

* $\lambda R(\alpha, \gamma)$ **(Tốc độ trung bình chuẩn hóa):**
    $$
    R(\alpha, \gamma) = \frac{1}{k} E[l(\gamma(\alpha(X)))]
    $$
    * $l(\gamma(i))$: Độ dài bit của codeword.
    * Mục đích là tối thiểu hóa chi phí bit trung bình trên mỗi mẫu. Trong trường hợp Tốc độ Biến thiên, $R$ được xấp xỉ bằng **Entropy của đầu ra** $H(q(X))$.

* $\lambda$ **(Hằng số Lagrange):** Tham số điều chỉnh trọng số giữa việc giảm Độ méo ($D$) và giảm Tốc độ ($R$).

---

### 5. Các kỹ thuật chủ chốt

1. **Thuật toán Lloyd (Generalized Lloyd Algorithm - GLA)**:
   * **Mô tả:** Phương pháp lặp để đạt được điều kiện tối ưu cục bộ. **Điều kiện Tối ưu bộ giải mã (Centroid Rule)** yêu cầu $y_i$ là trọng tâm (conditional expectation $E[X | X \in S_i]$) của vùng $S_i$ đối với MSE. **Điều kiện Tối ưu bộ mã hóa (Nearest Neighbor Rule)** yêu cầu $S_i$ là vùng Voronoi của $y_i$ (chọn $y_i$ gần $x$ nhất).

2. **Lý thuyết Tiệm cận Zador–Gersho (Asymptotic Theory)**:
   * **Mô tả:** Mở rộng công thức tiệm cận cho VQ $k$-chiều cố định tốc độ: $$\delta_k(R) \cong Z_k(R) = M_k \beta_k \sigma^2 2^{-2R}$$
   * **Ý nghĩa:** Lợi thế VQ đến từ $M_k$ (**Hằng số Gersho/Moment quán tính chuẩn hóa thấp nhất của hình dạng cell tối ưu**) và $\beta_k$ (**Yếu tố Zador/Sự phù hợp của mật độ điểm $\lambda(x)$ với $f(x)$**).

3. **Lượng tử hóa Tốc độ Biến thiên (Entropy-Constrained Quantization)**:
   * **Mô tả:** Tối ưu hóa bộ lượng tử hóa dưới ràng buộc Entropy đầu ra. Đối với tốc độ cao, bộ lượng tử hóa vô hướng **đồng nhất (uniform scalar quantization)** với mã hóa Entropy có hiệu suất gần tối ưu, chỉ kém giới hạn Shannon khoảng $1.53 \text{ dB}$.

4. **Vector Quantization (VQ) Cấu trúc**:
   * **Mô tả:** Các phương pháp giảm độ phức tạp $O(2^{kR})$ của VQ toàn bộ:
     * **Tree-Structured VQ (TSVQ):** Mã hóa bằng cách tìm kiếm theo cây nhị phân, giảm độ phức tạp xuống $O(kR)$.
     * **Multistage/Residual VQ:** Mã hóa phần dư (error) của giai đoạn trước ($\hat{X} = \hat{X}_1 + \hat{e}_2 + \dots$). Độ phức tạp tính toán và bộ nhớ là **tổng** của các giai đoạn.
     * **Lattice Quantization:** Hạn chế codebook là tập con của một lưới đều (ví dụ: $A_n, D_n, E_n$), cho phép mã hóa và giải mã nhanh với độ méo thấp cho các nguồn đồng nhất.

---

### 6. Các chỉ số đánh giá

* **Độ méo (Distortion, D)**:
  * **Mean Squared Error (MSE)**: $D(q) = E[||X - q(X)||^2]$.
  * **r-th Power Distortion**: $E[||X - q(X)||^r]$.
  * **Operational Distortion-Rate Function ($\delta(R)$)**: Độ méo nhỏ nhất đạt được với tốc độ $R$ hoặc ít hơn.

* **Tốc độ (Rate, R)**:
  * **Fixed Rate (Tốc độ cố định)**: $R(q) = \log_2 N$ (bits/sample).
  * **Entropy (Tốc độ biến thiên)**: $R(q) = H(q(X))$ (bits/sample).

* **Signal-to-Noise Ratio (SNR)**: $10 \log_{10} \frac{\text{var}(X)}{D(q)}$ (dB), thể hiện chất lượng tái tạo, thường được phân tích theo **quy tắc 6-dB-per-bit**.

* **Độ phức tạp (Complexity)**:
  * **Arithmetic Complexity (A)** (Tính toán).
  * **Storage Complexity (M)** (Bộ nhớ).
```eof