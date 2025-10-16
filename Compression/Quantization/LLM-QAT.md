### **Phân Tích Kỹ Thuật Bài Báo: LLM-QAT**

---

#### **1. Bài toán cần giải quyết**

* Bài báo giải quyết vấn đề suy giảm hiệu năng nghiêm trọng của các mô hình ngôn ngữ lớn (LLMs) khi lượng tử hóa xuống độ chính xác bit thấp (dưới 8-bit) bằng các phương pháp sau huấn luyện (post-training quantization), và đề xuất một phương pháp huấn luyện nhận biết lượng tử hóa (Quantization-Aware Training) để giải quyết vấn đề này.

---

#### **2. Đóng góp cốt lõi**

* Đóng góp cốt lõi là phương pháp **chưng cất kiến thức không cần dữ liệu (data-free knowledge distillation)**, sử dụng chính mô hình LLM gốc để tạo ra dữ liệu huấn luyện, qua đó lần đầu tiên áp dụng thành công Quantization-Aware Training (QAT) cho các LLM và đạt được lượng tử hóa 4-bit chính xác mà không cần truy cập vào bộ dữ liệu huấn luyện ban đầu.

---

#### **3. Quy trình của phương pháp (LLM-QAT)**

1.  **Bước 1: Tạo dữ liệu (Data Generation)**
    * Sử dụng mô hình gốc đã được huấn luyện (Pre-trained Model - Teacher) để tự sinh ra một bộ dữ liệu mới.
    * Quá trình này bắt đầu với một token ngẫu nhiên và sinh các token tiếp theo một cách lặp lại. Một chiến lược sinh "lai" (hybrid) được sử dụng: chọn các token đầu tiên một cách tất định (top-1) và sau đó lấy mẫu ngẫu nhiên từ phân phối đầu ra cho các token còn lại để tăng tính đa dạng.

2.  **Bước 2: Chưng cất kiến thức (Knowledge Distillation)**
    * Dữ liệu được tạo ra ở Bước 1 được dùng làm đầu vào cho cả mô hình Teacher (độ chính xác đầy đủ) và mô hình Student (mô hình đang được lượng tử hóa).
    * Mô hình Teacher tạo ra các "nhãn mềm" (soft labels), chính là phân phối xác suất trên toàn bộ từ vựng (logits).

3.  **Bước 3: Huấn luyện nhận biết lượng tử hóa (Quantization-Aware Training)**
    * Mô hình Student, với các trọng số (weights), hàm kích hoạt (activations), và bộ đệm KV (KV cache) được lượng tử hóa mô phỏng, được huấn luyện để tối thiểu hóa sự khác biệt giữa phân phối đầu ra của nó và các nhãn mềm từ mô hình Teacher thông qua hàm mất mát cross-entropy. Quá trình này giúp mô hình "học" cách hoạt động hiệu quả dưới các ràng buộc của lượng tử hóa.

---

#### **4. Hàm mục tiêu và Hàm mất mát**

* **Công thức:**
    Hàm mất mát chính được sử dụng là Cross-Entropy dựa trên chưng cất logits:
    $$
    \mathcal{L}_{CE}=-\frac{1}{n}\sum_{c}\sum_{i=1}^{n}p_{c}^{\mathcal{T}}(X_{i})log(p_{c}^{\mathcal{S}}(X_{i}))
    $$

* **Giải thích các thành phần:**
    * $\mathcal{L}_{CE}$: Giá trị của hàm mất mát Cross-Entropy.
    * $n$: Tổng số câu trong một batch huấn luyện.
    * $c$: Số lượng lớp, tương ứng với kích thước của bộ từ vựng (vocabulary size).
    * $p_{c}^{\mathcal{T}}(X_{i})$: Phân phối xác suất (nhãn mềm) được dự đoán bởi mô hình Teacher ($\mathcal{T}$) cho mẫu thứ $i$ ($X_i$). Đây là "kiến thức" cần được chưng cất.
    * $p_{c}^{\mathcal{S}}(X_{i})$: Phân phối xác suất được dự đoán bởi mô hình Student ($\mathcal{S}$) cho mẫu thứ $i$ ($X_i$).
    * **Mục đích:** Hàm mất mát này buộc phân phối đầu ra của mô hình Student lượng tử hóa phải bắt chước càng sát càng tốt phân phối đầu ra của mô hình Teacher có độ chính xác đầy đủ, giúp bảo toàn hiệu năng của mô hình gốc.

---

#### **5. Các kỹ thuật chủ chốt**

1.  **Chưng cất kiến thức không cần dữ liệu (Data-Free Knowledge Distillation):** Kỹ thuật nền tảng cho phép huấn luyện QAT mà không cần tập dữ liệu gốc, giải quyết bài toán về tính sẵn có và bảo mật của dữ liệu huấn luyện LLM.
2.  **Lượng tử hóa MinMax đối xứng (Symmetric MinMax Quantization):** Thay vì các phương pháp cắt bớt (clipping-based) hiện đại, nhóm tác giả nhận thấy việc giữ lại các giá trị ngoại lệ (outliers) là cực kỳ quan trọng đối với hiệu năng của LLM. Do đó, họ sử dụng phương pháp lượng tử hóa MinMax không cắt bớt để bảo toàn toàn bộ dải giá trị của trọng số và hàm kích hoạt.
3.  **Chiến lược sinh dữ liệu lai (Hybrid Data Generation Strategy):** Kết hợp việc chọn top-1 cho vài token đầu tiên để xác định xu hướng và lấy mẫu ngẫu nhiên cho các token sau để tăng tính đa dạng của dữ liệu sinh ra, giúp cải thiện hiệu quả huấn luyện.
4.  **Lượng tử hóa bộ đệm KV (KV Cache Quantization):** Ngoài trọng số và hàm kích hoạt, phương pháp này còn lượng tử hóa cả bộ đệm KV—một thành phần tiêu tốn nhiều bộ nhớ và là nút thắt cổ chai về thông lượng khi xử lý chuỗi dài.

---

#### **6. Các chỉ số đánh giá**

* **Perplexity:** Được sử dụng trên các tập dữ liệu WikiText2 và C4 để đánh giá khả năng mô hình ngôn ngữ bảo toàn được phân phối đầu ra của mô hình gốc.
* **Độ chính xác Zero-shot (Zero-shot Accuracy):** Đánh giá trên bộ 7 tác vụ Common Sense Reasoning (như BoolQ, PIQA, HellaSwag) để đo lường khả năng tổng quát hóa và suy luận của mô hình sau khi lượng tử hóa.
* **Độ chính xác Few-shot (Few-shot Accuracy):** Đánh giá trên các tập dữ liệu MMLU và TriviaQA để kiểm tra xem khả năng học trong ngữ cảnh (in-context learning) có được duy trì hay không.
* **Kích thước mô hình (Model Size):** Được đo bằng Gigabytes (GB) để thể hiện mức độ nén đạt được.