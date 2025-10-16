
***

### **Phân Tích Chuyên Sâu Bài Báo: LUT-GEMM**

**Tóm tắt chung:** Bài báo giới thiệu LUT-GEMM, một nhân (kernel) tính toán hiệu quả cho phép nhân ma trận lượng tử hóa, được thiết kế đặc thù để tăng tốc quá trình suy luận (inference) của các mô hình ngôn ngữ lớn (LLM). Cải tiến cốt lõi của phương pháp này là loại bỏ hoàn toàn bước giải lượng tử hóa (dequantization) tốn kém, vốn là một nút thắt cổ chai trong các phương pháp lượng tử hóa chỉ-trọng-số (weight-only quantization) trước đây.

---

### **Phần A: Bối cảnh và Sự cải tiến**

#### **1. Kế thừa từ đâu?**

Bài báo này xây dựng và cải tiến trực tiếp trên nền tảng của các công trình về **lượng tử hóa chỉ-trọng-số (weight-only quantization)** cho các mô hình ngôn ngữ lớn. Cụ thể, nó kế thừa từ:

* **Các phương pháp lượng tử hóa dưới 4-bit (sub-4-bit):** Các phương pháp như **OPTQ** và **AWQ** đã chứng minh tính hiệu quả của việc nén trọng số của LLM xuống 3 hoặc 4 bit trong khi vẫn giữ nguyên độ chính xác 16-bit (FP16) cho các giá trị kích hoạt (activations). Đây là bối cảnh chính mà LUT-GEMM hoạt động, sử dụng định dạng `W4/A16` (trọng số 4-bit, kích hoạt 16-bit).
* **Lượng tử hóa mã hóa nhị phân (Binary-Coding Quantization - BCQ):** LUT-GEMM sử dụng BCQ làm kỹ thuật lượng tử hóa nền tảng. BCQ biểu diễn một vector trọng số dưới dạng tổng có trọng số của nhiều vector nhị phân ($\{-1, +1\}$). Đây là chìa khóa để kiến trúc dựa trên Bảng tra cứu (LUT) có thể hoạt động.
* Link to BCQ [[XNOR-Net]]

#### **2. Điểm yếu của phương pháp cũ?**

Các phương pháp trước đây, dù đã thành công trong việc giảm dung lượng bộ nhớ, vẫn còn tồn tại những "nỗi đau" (pain points) về mặt hiệu suất tính toán mà LUT-GEMM nhắm đến để giải quyết:

* **Chi phí giải lượng tử hóa "on-the-fly":** Hạn chế lớn nhất của các phương pháp như OPTQ và AWQ là chúng yêu cầu một bước **giải lượng tử hóa (dequantization)** các trọng số từ 4-bit trở lại FP16 ngay trước khi thực hiện phép nhân ma trận. Quá trình này rất tốn tài nguyên và không làm giảm chi phí tính toán thực tế; nó chỉ giúp tăng tốc nhờ giảm băng thông di chuyển dữ liệu từ bộ nhớ.
* **Hạn chế của lượng tử hóa Integer 8-bit (INT8):** Việc lượng tử hóa cả trọng số và activation sang INT8 (W8A8) tuy có thể tận dụng các đơn vị số học integer chuyên dụng trên GPU nhưng lại gặp vấn đề. Việc lượng tử hóa activation có thể làm **giảm độ chính xác đáng kể** do sự xuất hiện của các giá trị ngoại lệ (outliers). Các kỹ thuật để khắc phục (như LLM.int8() hay SmoothQuant) thường phức tạp và yêu cầu phân tích đặc thù cho từng mô hình.

#### **3. Đóng góp mới là gì?**

Tác giả đã tuyên bố và chứng minh được các đóng góp mới và quan trọng sau:

1.  **Một nhân (kernel) nhân ma trận không cần giải lượng tử hóa:** Đóng góp cốt lõi là **LUT-GEMM**, một kernel cho phép thực hiện phép nhân ma trận trực tiếp trên các trọng số đã được lượng tử hóa, loại bỏ hoàn toàn bước dequantization. Điều này đạt được bằng cách sử dụng kỹ thuật tính toán dựa trên Bảng tra cứu (Look-Up Table - LUT).
2.  **Mở rộng định dạng BCQ để hỗ trợ cả lượng tử hóa đồng nhất (uniform):** Tác giả đã mở rộng định dạng BCQ truyền thống (vốn là bất đối xứng và không đồng nhất) bằng cách thêm vào một thành phần thiên vị (bias term `z`). Cải tiến đơn giản nhưng hiệu quả này cho phép BCQ có thể biểu diễn được cả các phương pháp lượng tử hóa đồng nhất, giúp LUT-GEMM có thể áp dụng cho một loạt các thuật toán nén mô hình hiện có.
3.  **Giới thiệu lượng tử hóa theo nhóm (group-wise quantization) linh hoạt:** Phương pháp này cho phép tạo ra một sự đánh đổi linh hoạt giữa tỷ lệ nén và độ chính xác bằng cách cho phép một hệ số tỷ lệ (scaling factor) được chia sẻ bởi một nhóm trọng số có kích thước tùy chỉnh `g`. Điều này tạo ra một không gian tìm kiếm cấu hình nén rộng hơn.

---

### **Phần B: Phân tích Kiến trúc và Thành phần mới**

#### **4. Cấu trúc tổng thể:**

LUT-GEMM không phải là một kiến trúc mô hình mới, mà là một **nhân tính toán (computational kernel)** thay thế cho phép toán nhân ma trận (GEMM - General Matrix Multiplication) tiêu chuẩn trong các LLM. Sơ đồ khối có thể được mô tả như sau:

1.  **Đầu vào:**
    * Ma trận trọng số `W` đã được lượng tử hóa theo định dạng BCQ (gồm các ma trận nhị phân `B` và các hệ số tỷ lệ `A`).
    * Vector kích hoạt `x` ở độ chính xác đầy đủ (FP16).
2.  **Khối 1: Tiền tính toán LUT (Pre-computation):**
    * Vector kích hoạt `x` được chia thành các vector con có độ dài `μ`.
    * Tất cả $2^\mu$ tổ hợp tuyến tính khả dĩ của các phần tử trong mỗi vector con được tính toán trước và lưu vào một Bảng tra cứu (LUT) trong bộ nhớ chia sẻ (shared memory) của GPU.
3.  **Khối 2: Tra cứu và Tích lũy (Lookup & Accumulation):**
    * Thay vì nhân và cộng, kernel sẽ duyệt qua các hàng của ma trận trọng số nhị phân `B`.
    * Mỗi chuỗi `μ` bit trong một hàng của `B` được sử dụng như một **chỉ số (index)** để tra cứu trực tiếp giá trị tổng riêng (partial sum) đã được tính toán trước từ LUT.
    * Các giá trị tra cứu được cộng dồn lại.
4.  **Khối 3: Nhân hệ số và Tổng hợp (Scaling & Aggregation):**
    * Kết quả tổng hợp từ bước tra cứu được nhân với các hệ số tỷ lệ (scaling factors) tương ứng trong ma trận `A`.
5.  **Đầu ra:**
    * Vector đầu ra `y`, tương đương với kết quả của `W * x`, ở độ chính xác FP16.

#### **5. Các khối xây dựng (Building Blocks):**

Các thành phần chính tạo nên cơ chế hoạt động của LUT-GEMM bao gồm:

* **Biểu diễn BCQ mở rộng:** Nền tảng toán học cho phép biểu diễn trọng số dưới dạng $\tilde{w}=\sum_{i=0}^{q-1}(\alpha_{i} \cdot b_{i})+z$.
* **Cơ chế tạo LUT:** Một module tính toán các tổng riêng của vector kích hoạt và lưu chúng vào bộ nhớ nhanh.
* **Lõi tra cứu (Lookup Core):** Thay thế các phép toán số học bằng các thao tác truy cập bộ nhớ (memory access) dựa trên chỉ số là các bit trọng số.
* **Chiến lược song song hóa trên GPU:** Phân chia ma trận lớn thành các khối nhỏ (tile), mỗi khối được xử lý bởi một Thread Block (TB) trên GPU. Các LUT được lưu trong **Shared Memory** để tăng tốc độ truy cập và tái sử dụng giữa các thread trong cùng một TB.

#### **6. Thành phần "ăn tiền" (Novel Component):**

Thành phần mới lạ và cốt lõi nhất chính là **sự kết hợp giữa biểu diễn BCQ và cơ chế tính toán dựa trên LUT để loại bỏ hoàn toàn phép nhân số học trong quá trình xử lý ma trận nhị phân.**

* **Cấu tạo và cách hoạt động:**
    1.  **Vấn đề:** Phép nhân ma trận nhị phân `B` với vector kích hoạt `x` (ví dụ `Bx^T`) chứa rất nhiều phép tính lặp lại. Chẳng hạn, một tổ hợp như `(x₁ + x₂ - x₃)` có thể xuất hiện nhiều lần nếu các hàng tương ứng của `B` có cùng mẫu bit `[+1, +1, -1]`.
    2.  **Giải pháp của LUT-GEMM:** Thay vì tính toán lại nhiều lần, ta hãy tính trước tất cả các kết quả có thể.
    3.  **Ví dụ:** Giả sử ta chọn kích thước cửa sổ (sub-vector length) `μ=3`. Vector kích hoạt `x` được chia thành các nhóm 3 phần tử. Với nhóm `(x₁, x₂, x₃)`, ta sẽ tính trước tất cả $2^3=8$ khả năng: `±x₁ ± x₂ ± x₃` và lưu vào một LUT.
    4.  **Thực thi:** Khi xử lý một hàng của ma trận `B`, ta lấy ra 3 bit, ví dụ `[+1, +1, -1]`. Chuỗi bit này tương ứng với một chỉ số (ví dụ, `110` trong hệ nhị phân). Ta dùng chỉ số này để lấy ra giá trị `(x₁ + x₂ - x₃)` đã được tính sẵn trong LUT chỉ bằng một thao tác đọc bộ nhớ duy nhất.
    5.  **Lợi ích:** Phép toán này biến `μ-1` phép cộng/trừ thành **một phép truy cập bộ nhớ duy nhất**. Khi LUT được đặt trong shared memory cực nhanh của GPU, tốc độ được cải thiện đáng kể so với việc thực hiện các phép toán số học truyền thống.

---

### **Phần C: Quy trình hoạt động (Pipeline)**

#### **7. Pipeline Huấn luyện (Training Pipeline):**

Bài báo này tập trung vào **Lượng tử hóa sau huấn luyện (Post-Training Quantization - PTQ)**, nghĩa là mô hình đã được huấn luyện xong. Do đó, "pipeline huấn luyện" ở đây thực chất là **pipeline lượng tử hóa**.

* **Input:** Một mô hình LLM đã được huấn luyện với các trọng số ở độ chính xác cao (FP16 hoặc FP32).
* **Step 1: Lượng tử hóa trọng số:**
    * Với mỗi lớp tuyến tính (linear layer) trong mô hình, ma trận trọng số `W` được đưa qua một thuật toán tối ưu hóa (ví dụ, solver được giới thiệu trong OPTQ).
    * Mục tiêu của thuật toán này là tìm ra các ma trận nhị phân `B_i`, hệ số tỷ lệ `α_i` và bias `z` sao cho sai số lượng tử hóa (ví dụ, `||W - W_quantized||`) là nhỏ nhất.
* **Step 2: Không có hàm mất mát và lan truyền ngược:** Quá trình này không cập nhật lại trọng số của mô hình thông qua backpropagation. Đây là một quá trình chuyển đổi định dạng trọng số.
* **Output:** Một mô hình LLM mới với tất cả các trọng số đã được chuyển đổi sang định dạng BCQ mở rộng và sẵn sàng cho quá trình suy luận bằng kernel LUT-GEMM.

#### **8. Pipeline Suy luận (Inference Pipeline):**

Đây là giai đoạn mà LUT-GEMM phát huy tác dụng.

* **Input:** Một chuỗi văn bản đầu vào (prompt).
* **Step 1: Tiền xử lý:** Chuỗi văn bản được token hóa và chuyển thành các vector embedding FP16 như một LLM thông thường.
* **Step 2: Forward Pass với LUT-GEMM:**
    * Dữ liệu đi qua các lớp của mô hình.
    * Tại mỗi lớp tuyến tính, thay vì dùng kernel GEMM tiêu chuẩn, hệ thống sẽ gọi **kernel LUT-GEMM**.
    * Kernel này sẽ thực hiện quy trình 5 bước đã mô tả trong Phần B.4: (1) Nhận trọng số BCQ và activation FP16 -> (2) Tiền tính toán LUT -> (3) Tra cứu và tích lũy -> (4) Nhân hệ số -> (5) Trả về kết quả FP16.
    * Các phép toán khác như self-attention, layer normalization, v.v., vẫn được thực hiện ở độ chính xác FP16.
* **Step 3: Sinh token:** Mô hình tạo ra một phân phối xác suất trên kho từ vựng và sinh ra token tiếp theo. Token này được đưa trở lại làm đầu vào cho bước tiếp theo trong quá trình sinh tự hồi quy (auto-regressive generation).
* **Khác biệt so với lúc "huấn luyện" (lượng tử hóa):**
    * **Không có dequantization:** Đây là điểm khác biệt cốt lõi và là ưu thế lớn nhất so với các phương pháp như OPTQ/AWQ, vốn phải giải lượng tử hóa trọng số về FP16 trước khi nhân. LUT-GEMM hoạt động trực tiếp trên định dạng nén.
    * Các kỹ thuật tối ưu hóa suy luận tiêu chuẩn khác (ví dụ: không sử dụng dropout) cũng được áp dụng.