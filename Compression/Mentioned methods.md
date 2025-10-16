## 4. Pruning (Cắt tỉa)

### 4.1 Unstructured Pruning (Cắt tỉa không cấu trúc)

| Phương pháp | Trích dẫn (Tác giả và Năm)   |
| ----------- | ---------------------------- |
| SparseGPT   | Frantar and Alistarh, 2023 1 |
| Wanda       | Sun et al., 2024 2           |
| SAMSP       | Shao et al., 2024a 3         |
| DSnoT       | Zhang et al., 2024 4         |
| Flash-LLM   | Xia et al., 2023 5           |

---

### 4.2 Structured Pruning (Cắt tỉa có cấu trúc)

#### Loss-based Pruning (Cắt tỉa dựa trên hàm lỗi)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|LLM-Pruner|Ma et al., 2023 6|
|Shortened LLAMA|Kim et al., 2024 7|

#### Magnitude-based Pruning (Cắt tỉa dựa trên độ lớn)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|FLAP|An et al., 2024 8|
|SliceGPT|Ashkboos et al., 2024 9|

#### Regularization-based Pruning (Cắt tỉa dựa trên Điều chuẩn)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|Sheared LLAMA|Xia et al., 2024 10|

---

### 4.3 Semi-structured Pruning (Cắt tỉa bán cấu trúc)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|E-Sparse|Li et al., 2023b 11|
|SparseGPT|Frantar and Alistarh, 2023 12|
|Wanda|Sun et al., 2024 13|
|(2:4 fine-grained sparsity)|Choquette et al., 2021 14|

## 5. Knowledge Distillation (Chưng cất Tri thức)

### 5.1 Black-box KD (Chưng cất Tri thức Hộp đen)

#### 5.1.1 Chain-of-Thought (CoT) Distillation (Chưng cất Chuỗi suy nghĩ)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|MT-COT|Li et al., 2024b|
|CoT Prompting|Magister et al., 2023|
|Fine-tune-CoT|Ho et al., 2023|
|SSLM|Fu et al., 2023|
|SCOTT|Wang et al., 2023a|
|Distilling Step-by-Step|Hsieh et al., 2023|
|SOCRATIC COT|Shridhar et al., 2023|
|PaD|Zhu et al., 2024|
|DRA|Wang et al., 2023f|
|TDIG|Li et al., 2024c|

#### 5.1.2 In-Context Learning (ICL) Distillation (Chưng cất Học tập trong Ngữ cảnh)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|In-context Learning Distillation|Huang et al., 2022|
|AICD|Liu, 2024|

#### 5.1.3 Instruction Following (IF) Distillation (Chưng cất Tuân thủ Hướng dẫn)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|Lion|Jiang et al., 2023|
|LaMini-LM|Wu et al., 2024|
|SELF-INSTRUCT|Wang et al., 2023d|
|Selective Reflection-Tuning|Li et al., 2024a|

---

### 5.2 White-box KD (Chưng cất Tri thức Hộp trắng)

|Phương pháp|Trích dẫn (Tác giả và Năm)|
|---|---|
|MINILLM|Gu et al., 2024|
|GKD|Agarwal et al., 2024|
|TED|Liang et al., 2023|
## 6: Low-Rank Factorization (Phân tích Thừa số Hạng Thấp)**

| Phương pháp | Trích dẫn (Tác giả và Năm) | Mô tả chi tiết                                                                                                                                                                                                                                                                                                         |
| ----------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LPLR**    | Saha et al., 2023 2        | Nén ma trận trọng số của LLMs thông qua **phân tích thừa số hạng thấp ngẫu nhiên và độ chính xác thấp**3. Nó xấp xỉ không gian cột bằng kỹ thuật phác thảo ngẫu nhiên và tạo ra hai thừa số hạng thấp được lưu trữ ở độ chính xác thấp4.                                                                               |
| **ASVD**    | Yuan et al., 2023b 5       | Sử dụng **Phân tích Giá trị Số ít (SVD)** nhận biết kích hoạt6. Nó **chia tỷ lệ ma trận trọng số** bằng một ma trận đường chéo dựa trên phân phối kích hoạt của các kênh tính năng đầu vào7. ASVD cũng gán tỷ lệ nén tối ưu cho các lớp khác nhau bằng cách phân tích phân phối giá trị số ít8.                        |
| **LASER**   | Sharma et al., 2024 9      | Áp dụng **Giảm hạng Chọn lọc Lớp** (Layer-Selective Rank Reduction)10. Nó liên quan đến việc giảm hạng có chọn lọc các thành phần bậc cao hơn của ma trận trọng số11. Kỹ thuật này được chứng minh là cải thiện khả năng xử lý dữ liệu huấn luyện hiếm và khả năng chống lại việc diễn giải lại câu hỏi của mô hình12. |