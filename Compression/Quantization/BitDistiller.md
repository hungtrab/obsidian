- Code: 
***

### **Phân Tích Kỹ Thuật Bài Báo: BitDistiller**

#### 1.  **Bài toán cần giải quyết**
* Bài báo này giải quyết vấn đề suy giảm hiệu năng nghiêm trọng của các Mô hình Ngôn ngữ Lớn (LLMs) khi lượng tử hóa trọng số xuống các mức độ chính xác cực thấp (dưới 4-bit) bằng cách cải thiện độ trung thực của trọng số và tối ưu hóa quá trình học biểu diễn ở mức bit thấp.

---
#### 2.  **Đóng góp cốt lõi**
* Đóng góp mới lạ và quan trọng nhất là việc đề xuất hàm mục tiêu **Phân kỳ Kullback-Leibler Nhận thức về Độ tin cậy (Confidence-Aware Kullback-Leibler Divergence - CAKLD)**, một cơ chế chưng cất tri thức (knowledge distillation) có khả năng tự động cân bằng giữa hai hành vi "tìm kiếm mode" (mode-seeking) và "phủ mode" (mode-covering) dựa trên độ tin cậy của mô hình thầy (teacher model).

---
#### 3.  **Quy trình của phương pháp (BitDistiller)**
Quy trình được mô tả trong Thuật toán 1:
* **Bước 1: Khởi tạo:** Áp dụng kỹ thuật **cắt xén bất đối xứng (asymmetric clipping)** một lần duy nhất lên các trọng số đầy đủ độ chính xác (full-precision weights) để xử lý các giá trị ngoại lai và tạo điểm khởi đầu tốt cho quá trình huấn luyện.
* **Bước 2: Vòng lặp huấn luyện QAT (Quantization-Aware Training):**
    * Trong mỗi bước huấn luyện, thực hiện lượt truyền xuôi (forward pass) với các trọng số đã được lượng tử hóa một cách linh động ($w_{Q}^{t}=Q(w^{t})$).
    * Tính toán giá trị mất mát giữa mô hình thầy (trọng số đầy đủ độ chính xác $w$) và mô hình trò (trọng số đã lượng tử hóa $w_{Q}^{t}$) bằng hàm mất mát **CAKLD**.
* **Bước 3: Cập nhật trọng số:** Thực hiện lượt truyền ngược (backward pass) và cập nhật gradient lên các **trọng số đầy đủ độ chính xác** ($w^{t}$), không phải trọng số đã lượng tử hóa.
* **Bước 4: Đầu ra:** Sau khi quá trình huấn luyện kết thúc, các trọng số đầy đủ độ chính xác cuối cùng ($w^{T}$) sẽ được lượng tử hóa một lần nữa để cho ra mô hình nén cuối cùng ($w_{Q}^{T}$).

---
#### 4.  **Hàm mục tiêu và Hàm mất mát**
* **Công thức:** Hàm mục tiêu chính là **CAKLD**, được định nghĩa trong Công thức (5):
    $$\mathcal{D}_{CAKLD}(P_{T}||P_{S}) = \gamma\mathcal{D}_{KL}(P_{S}||P_{T}) + (1-\gamma)\mathcal{D}_{KL}(P_{T}||P_{S})$$
* **Giải thích các thành phần:**
    * $P_{T}$ và $P_{S}$: Tương ứng là phân phối xác suất đầu ra của mô hình **Thầy** (Teacher - full-precision) và mô hình **Trò** (Student - quantized).
    * $\mathcal{D}_{KL}(P_{S}||P_{T})$: **Phân kỳ KL Nghịch (Reverse KL)**, có hành vi **"tìm kiếm mode" (mode-seeking)**. Thành phần này buộc mô hình Trò phải tái tạo chính xác các đầu ra có xác suất cao (đỉnh của phân phối) của mô hình Thầy.
    * $\mathcal{D}_{KL}(P_{T}||P_{S})$: **Phân kỳ KL Thuận (Forward KL)**, có hành vi **"phủ mode" (mode-covering)**. Thành phần này khuyến khích mô hình Trò bao phủ toàn bộ phân phối của mô hình Thầy, kể cả các đầu ra có xác suất thấp, giúp đa dạng hóa và giữ lại nhiều thông tin hơn.
    * $\gamma$: **Hệ số nhận thức về độ tin cậy**, được tính toán dựa trên xác suất token trung bình của mô hình Thầy trên dữ liệu huấn luyện. Khi mô hình Thầy tự tin (giá trị $\gamma$ cao), hàm mất mát sẽ ưu tiên KL Nghịch (mode-seeking). Ngược lại, khi mô hình Thầy không chắc chắn ($\gamma$ thấp), nó sẽ ưu tiên KL Thuận (mode-covering).

---
#### 5.  **Các kỹ thuật chủ chốt**
1.  **Confidence-Aware KLD (CAKLD):** Kỹ thuật cốt lõi, là một hàm mất mát lai (hybrid loss) tự động điều chỉnh chiến lược chưng cất tri thức, giúp mô hình linh hoạt hơn trên nhiều loại tác vụ khác nhau (ví dụ: tác vụ lý luận đòi hỏi độ chính xác cao so với tác vụ sinh văn bản đòi hỏi sự đa dạng).
2.  **Lượng tử hóa và Cắt xén Bất đối xứng (Asymmetric Quantization and Clipping):** Thay vì dùng phương pháp đối xứng, BitDistiller sử dụng lượng tử hóa bất đối xứng để phù hợp hơn với phân phối tự nhiên của trọng số LLMs. Đặc biệt, kỹ thuật cắt xén bất đối xứng chỉ được áp dụng một lần khi khởi tạo để giảm chi phí tính toán mà vẫn mang lại hiệu quả cao.
3.  **Chưng cất Tự thân (Self-Distillation):** Khung chưng cất tri thức sử dụng chính mô hình đầy đủ độ chính xác làm "thầy" để dạy cho phiên bản lượng tử hóa của nó. Điều này đảm bảo sự tương thích hoàn hảo về kiến trúc và có thể giúp việc căn chỉnh trọng số và phân phối xác suất hiệu quả hơn.
4.  **Dữ liệu do Thầy tạo ra (Teacher-Generated Data):** Thay vì dùng dữ liệu gốc (ground truth), phương pháp này cho thấy hiệu quả vượt trội khi huấn luyện với dữ liệu do chính mô hình Thầy sinh ra ($y_p$). Dữ liệu này có phân phối logit với độ tin cậy cao, giúp quá trình hội tụ với CAKLD trở nên dễ dàng và hiệu quả hơn.

---
#### 6.  **Các chỉ số đánh giá**
Họ đã sử dụng một loạt các chỉ số tiêu chuẩn để đánh giá trên nhiều phương diện:
* **Mô hình hóa ngôn ngữ (Language Modeling):**
    * **Perplexity (PPL):** Đo lường mức độ "ngạc nhiên" của mô hình trên tập dữ liệu WikiText-2, PPL càng thấp càng tốt.
* **Hiểu ngôn ngữ và kiến thức chung (General Language Understanding):**
    * **MMLU (5-shot):** Đánh giá khả năng học trong ngữ cảnh (in-context learning) trên nhiều lĩnh vực kiến thức.
    * **PIQA, HellaSwag, WinoGrande, ARC-c:** Các bộ benchmark về suy luận thường thức (common sense reasoning).
* **Các tác vụ lý luận phức tạp (Complex Reasoning):**
    * **HumanEval:** Đánh giá khả năng sinh mã lập trình (code generation).
    * **GSM8K:** Đánh giá khả năng giải các bài toán đố (mathematical reasoning).