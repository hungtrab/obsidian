## Tổng quan về Nén Mô hình cho LLMs

Các **Mô hình Ngôn ngữ Lớn (LLMs)** đã cách mạng hóa lĩnh vực xử lý ngôn ngữ tự nhiên, nhưng kích thước lớn và nhu cầu tính toán cao của chúng tạo ra những thách thức đáng kể cho việc sử dụng thực tế, đặc biệt là trong các môi trường tài nguyên hạn chế. Ví dụ, mô hình GPT-175B với 175 tỷ tham số yêu cầu tối thiểu 350GB bộ nhớ ở định dạng độ chính xác bán phần (FP16).

**Nén mô hình** là một lĩnh vực nghiên cứu trọng điểm nhằm giải quyết những thách thức này. Nó bao gồm việc biến đổi một mô hình lớn, tốn tài nguyên thành một phiên bản nhỏ gọn phù hợp để triển khai trên các thiết bị bị hạn chế về tài nguyên. Ngoài ra, nén mô hình có thể **tăng tốc độ suy luận (inference speed)** của LLM và tối ưu hóa **hiệu quả tài nguyên**.

Bốn loại kỹ thuật nén mô hình chính được khảo sát là:

1. **Quantization (Lượng tử hóa)**
    
2. **Pruning (Cắt tỉa)**
    
3. **Knowledge Distillation (Chưng cất tri thức)**
    
4. **Low-Rank Factorization (Phân tích thừa số hạng thấp)**

---
## 3. Quantization (Lượng tử hóa)

[[Quantization]]
**Ý chính dẫn dắt:**

**Lượng tử hóa** (Quantization) là quá trình **giảm số lượng bit** (tức là độ chính xác) trong các tham số của mô hình với mục tiêu **giảm thiểu tổn thất hiệu suất suy luận**. 

Lượng tử hóa được phân loại thành hai cách tiếp cận chính: **Quantization-Aware Training (QAT)** và **Post-Training Quantization (PTQ)**. Sự khác biệt cơ bản là liệu có cần **huấn luyện lại** trong quá trình lượng tử hóa hay không.

| Category | Methods      | LLM        | Bit Width (Weights, Activations, KV Cache) | Perplexity Difference (Wikitext-2, C4) | Speedup |
| -------- | ------------ | ---------- | ------------------------------------------ | -------------------------------------- | ------- |
| QAT      | LLM-QAT      | LLAMA-30B  | 4, 8, 16                                   | 0.9                                    | -       |
|          | BitDistiller | LLAMA2-13B | 2, 16, 16                                  | 0.5                                    | -       |
|          | OneBit       | LLaMA-13B  | 1, 16, 16                                  | 1.9                                    | -       |

<hr>
### 3.1 Quantization-Aware Training (QAT - Huấn luyện nhận biết Lượng tử hóa) 

[[Quantization-Aware Training]]
**Ý chính dẫn dắt:**

**QAT** bao gồm việc **huấn luyện lại** một mô hình đã được lượng tử hóa để **khắc phục sự suy giảm hiệu suất** do lượng tử hóa gây ra.

**Các ví dụ đại diện:**

- **LLM-QAT**: Triển khai khung QAT tiêu chuẩn trực tiếp trên LLMs. Nó **chưng cất tri thức** bằng cách tự tạo dữ liệu từ LLM gốc và huấn luyện LLM được lượng tử hóa để **căn chỉnh với phân phối đầu ra** của mô hình gốc dựa trên dữ liệu đã tạo.
    
- **BitDistiller**: **Hợp nhất QAT với tự chưng cất** (self-distillation), tăng cường hiệu suất của LLM ở độ chính xác **dưới 4-bit**. Nó sử dụng lượng tử hóa **bất đối xứng** được điều chỉnh, **cắt xén** (clipping), và mục tiêu **phân kỳ Kullback-Leibler Nhận biết Độ tin cậy** (Confidence-Aware Kullback-Leibler Divergence objective) để hội tụ nhanh hơn và đạt kết quả vượt trội.
    
- **OneBit**: Giới thiệu một phương pháp biểu diễn tham số **1-bit** và phương pháp khởi tạo tham số hiệu quả để triển khai lượng tử hóa **1-bit** cho ma trận trọng số LLM, mở đường cho việc triển khai độ rộng bit cực thấp.
    

#### Remark 1 về QAT và PEFT

- Mặc dù QAT có thể giảm thiểu sự suy giảm độ chính xác, nhưng việc **huấn luyện lại đòi hỏi rất nhiều nỗ lực** do LLMs có hàng chục hoặc hàng trăm tỷ tham số.
    
- Một giải pháp thực tế là **kết hợp Parameter-Efficient Fine-Tuning (PEFT)** vào quá trình huấn luyện lại QAT.
    
- Các phương pháp như **QLORA**, **PEQA** và **LoftQ** đã kết hợp lượng tử hóa với PEFT để tăng hiệu quả tinh chỉnh mô hình, tuy nhiên chúng thường **phụ thuộc vào nhiệm vụ** (task-dependent). **L4Q** là một nỗ lực sơ bộ nhằm tăng cường tính tổng quát bằng cách tận dụng kích thước bước lượng tử hóa học được theo kiểu LoRA cho LLMs.
    

<hr>

### 3.2 Post-Training Quantization (PTQ - Lượng tử hóa sau huấn luyện)

[[Post-Training Quantization]]
**Ý chính dẫn dắt:**

**PTQ** chuyển đổi một LLM độ chính xác đầy đủ sang độ chính xác thấp **mà không cần huấn luyện lại**, giúp **tiết kiệm bộ nhớ và chi phí tính toán**.

PTQ được phân loại thành ba nhóm: **Weight-Only Quantization** (Lượng tử hóa chỉ trọng số), **Weight-Activation Quantization** (Lượng tử hóa trọng số-kích hoạt), và **KV Cache Quantization** (Lượng tử hóa bộ nhớ đệm KV). Sự khác biệt nằm ở **mục tiêu lượng tử hóa**.

#### 3.2.1 Weight-Only Quantization (Lượng tử hóa chỉ trọng số)

**Ý chính dẫn dắt:**

Đây là phương pháp thông thường và phổ biến nhất, **chỉ tập trung vào lượng tử hóa trọng số**. Lượng tử hóa chỉ trọng số thường có thể đạt được độ rộng bit thấp hơn vì lượng tử hóa kích hoạt nhạy cảm hơn. Tuy nhiên, nó có thể **tạo ra chi phí tính toán bổ sung** trong quá trình suy luận do cần khử lượng tử hóa trước khi nhân ma trận.

**Các ví dụ đại diện:**

- **LUT-GEMM**: Sử dụng định dạng lượng tử hóa mã hóa nhị phân (BCQ) để **phân tích thừa số tham số** của LLMs thành các tham số nhị phân và một tập hợp các hệ số chia tỷ lệ, nhằm **tăng tốc phép nhân ma trận** được lượng tử hóa.
    
- **GPTQ**: Đề xuất phương pháp lượng tử hóa **từng lớp** dựa trên Tối ưu hóa Lượng tử hóa Bộ não (OBQ), cập nhật trọng số bằng thông tin Hessian đảo ngược, lượng tử hóa LLMs thành **3/4-bit**.
    
- **QuIP**: Điều chỉnh trọng số tối ưu bằng cách sử dụng phép phân tích **LDL** của ma trận Hessian và nhân ma trận trọng số/Hessian với tích Kronecker của các ma trận trực giao ngẫu nhiên để đảm bảo tính **không cố kết** (incoherence). QuIP lượng tử hóa thành **2-bit** với tổn thất hiệu suất tối thiểu.
    
- **AWQ** : Lưu trữ **1% trọng số hàng đầu** có tác động đáng kể nhất đến hiệu suất của LLM ở **độ chính xác cao** (phát hiện trọng số nhạy cảm) và tích hợp phương pháp chia tỷ lệ trên mỗi kênh (per-channel scaling) để xác định các hệ số chia tỷ lệ tối ưu.
    
- **OWQ**: Tương tự AWQ, nó lưu trữ các trọng số **nhạy cảm với các outlier kích hoạt** ở độ chính xác cao, và lượng tử hóa các trọng số không nhạy cảm khác.
    
- **SpQR**: Sử dụng **lỗi $\Large L_2$** giữa các dự đoán gốc và đã lượng tử hóa làm metric độ nhạy trọng số.
    
- **SqueezeLLM**: Giới thiệu thuật toán **cụm trọng số** (weights clusters) dựa trên độ nhạy (ước tính bằng ma trận Hessian), sử dụng tâm k-means làm giá trị trọng số được lượng tử hóa. Nó lượng tử hóa LLMs thành **3-bit**, đạt được tốc độ tăng tốc hơn **2 lần** so với chuẩn FP16.
    

#### 3.2.2 Weight-Activation Quantization (Lượng tử hóa Trọng số-Kích hoạt)

**Ý chính dẫn dắt:**

Mở rộng mục tiêu lượng tử hóa sang **cả trọng số và kích hoạt**. LLMs có **các outlier trong kích hoạt**, và hiệu suất suy giảm nhiều nếu các kích hoạt này được lượng tử hóa trực tiếp. Các công trình gần đây cố gắng xử lý đặc biệt các outlier này để giảm lỗi lượng tử hóa.

**Các ví dụ đại diện:**

- **ZeroQuant**: Là công trình đầu tiên thực hiện lượng tử hóa trọng số-kích hoạt cho LLMs, sử dụng lượng tử hóa **theo nhóm** (group-wise) cho trọng số và **theo token** (token-wise) cho kích hoạt, giảm độ chính xác xuống **INT8** cho cả hai.
    
- **LLM.int8()**: Lưu trữ các chiều tính năng outlier ở **độ chính xác cao** và sử dụng **lượng tử hóa theo vector** (vector-wise quantization) — gán hằng số chuẩn hóa riêng cho mỗi tích bên trong phép nhân ma trận — để lượng tử hóa các tính năng khác. Nó lượng tử hóa thành **8-bit** mà **không suy giảm hiệu suất**.
    
- **Smooth Quant**: Thiết kế một phép biến đổi **chia tỷ lệ trên mỗi kênh** để **làm mịn các outlier kích hoạt** dựa trên việc phát hiện rằng các token khác nhau có sự thay đổi tương tự nhau giữa các kênh kích hoạt.
    
- **RPTQ**: Tích hợp phương pháp **sắp xếp lại kênh** (channel reordering), nhóm và sắp xếp lại các kênh trong kích hoạt để sử dụng cùng một tham số lượng tử hóa cho các giá trị trong mỗi cụm, nhằm giảm hiệu ứng của sự khác biệt về phạm vi số giữa các kênh.
    
- **Olive**: Sử dụng lượng tử hóa **cặp outlier-victim (OVP)** để xử lý các giá trị outlier cục bộ với chi phí phần cứng thấp.
    
- **LLM-FP4**: Sử dụng **định dạng dấu phẩy động** (cụ thể là FP8 và **FP4**) để giải quyết các giới hạn của lượng tử hóa số nguyên truyền thống khi đối phó với các outlier. Nó cũng giới thiệu một khung tìm kiếm để xác định **độ lệch số mũ tối ưu** và **giá trị lượng tử hóa tối đa**.
    
- **OmniQuant**: Xử lý các outlier kích hoạt bằng cách **chuyển thách thức lượng tử hóa từ kích hoạt sang trọng số** một cách tương đương, và tối ưu hóa **ngưỡng cắt** để điều chỉnh các giá trị cực trị của trọng số.
    

#### 3.2.3 KV Cache Quantization (Lượng tử hóa Bộ nhớ đệm KV)

**Ý chính dẫn dắt:**

Với số lượng token đầu vào được hỗ trợ tăng lên, mức sử dụng bộ nhớ của **KV cache** cũng tăng theo, trở thành nút thắt cổ chai về bộ nhớ. Lượng tử hóa KV cache nhằm mục đích **giảm dung lượng bộ nhớ** và **tăng tốc suy luận**.

**Các ví dụ đại diện:**

- **KVQuant** 63: Đề xuất một số phương pháp Lượng tử hóa KV Cache, chẳng hạn như Lượng tử hóa Key trên mỗi Kênh (Per-Channel Key Quantization), Lượng tử hóa Key PreROPE, và lượng tử hóa KV cache Bất đối xứng, để triển khai suy luận LLM với độ dài ngữ cảnh **10 triệu**.
    
- **KIVI** 65: Qua phân tích chuyên sâu, KIVI phát hiện **cache khóa** nên được lượng tử hóa **trên mỗi kênh** (per-channel), trong khi **cache giá trị** nên được lượng tử hóa **trên mỗi token** (per-token)66. KIVI lượng tử hóa KV cache xuống **2 bit** mà **không cần tinh chỉnh**.
    
- **WKVQuant** 68: Trình bày cách tiếp cận để **lượng tử hóa cả trọng số và KV cache**69. Nó tích hợp lượng tử hóa chỉ quá khứ (past-only quantization) để tinh chỉnh các phép tính chú ý, sử dụng chiến lược lượng tử hóa **hai chiều** để quản lý phân phối KV cache, và sử dụng điều chuẩn tái tạo chéo khối (cross-block reconstruction regularization) để tối ưu hóa tham số70.

---

## 4. Pruning (Cắt tỉa)

[[Pruning]]
**Ý chính dẫn dắt:**

Cắt tỉa là một kỹ thuật mạnh mẽ nhằm **giảm kích thước hoặc độ phức tạp của mô hình bằng cách loại bỏ các thành phần dư thừa**.

Cắt tỉa được chia thành ba loại: **Unstructured Pruning** (Cắt tỉa không cấu trúc), **Structured Pruning** (Cắt tỉa có cấu trúc), và **Semi-structured Pruning** (Cắt tỉa bán cấu trúc).

### 4.1 Unstructured Pruning (Cắt tỉa không cấu trúc)

**Ý chính dẫn dắt:**

Cắt tỉa không cấu trúc loại bỏ **các tham số riêng lẻ** , dẫn đến một cấu trúc thưa không đều. Các phương pháp thường **bỏ qua việc huấn luyện lại** để phục hồi hiệu suất, nhưng mô hình thưa không đều cần **xử lý chuyên biệt hoặc tối ưu hóa phần mềm** để tăng tốc suy luận.

**Các ví dụ đại diện:**

- **SparseGPT**: Giới thiệu chiến lược cắt tỉa **một lần (one-shot)** mà không cần huấn luyện lại7. Nó coi việc cắt tỉa là một bài toán hồi quy thưa rộng lớn 8, đạt được độ thưa không cấu trúc đáng kể (lên đến hơn 50% trên các mô hình GPT lớn)9.
    
- **Wanda**: Đạt được độ thưa bằng cách cắt tỉa các trọng số có **độ lớn nhỏ nhất** nhân với chuẩn (norm) của các kích hoạt đầu vào tương ứng, **mà không cần huấn luyện lại hoặc cập nhật trọng số**10.
    
- **SAMSP**: Sử dụng **ma trận Hessian** làm metric để đánh giá độ nhạy của ma trận trọng số và **điều chỉnh động sự phân bổ độ thưa** dựa trên độ nhạy11.
    
- **Flash-LLM**: Giới thiệu phương pháp nhân ma trận thưa không cấu trúc để cung cấp **hỗ trợ phần cứng** cho cắt tỉa không cấu trúc trên GPU Tensor Core12.
    

---

### 4.2 Structured Pruning (Cắt tỉa có cấu trúc)

**Ý chính dẫn dắt:**

Cắt tỉa có cấu trúc loại bỏ **toàn bộ các thành phần** (như nơ-ron, đầu chú ý, hoặc lớp) 13, mang lại lợi thế là **không phụ thuộc vào phần cứng** và cho phép suy luận tăng tốc trên phần cứng truyền thống14. Tuy nhiên, việc loại bỏ các thành phần lớn hơn có thể dẫn đến **suy giảm hiệu suất**, thường đòi hỏi **tinh chỉnh tham số hiệu quả** để phục hồi15.

Các công trình được chia thành ba nhóm dựa trên metric cắt tỉa: **Loss-based Pruning** (Cắt tỉa dựa trên hàm lỗi), **Magnitude-based Pruning** (Cắt tỉa dựa trên độ lớn), **Regularization-based Pruning** (Cắt tỉa dựa trên Điều chuẩn)16.

**Các ví dụ đại diện:**

- **Loss-based Pruning**:
    
    - **LLM-Pruner**: Giới thiệu cắt tỉa cấu trúc một lần dựa trên **thông tin gradient**17. Nó xác định các cấu trúc phụ thuộc và chọn nhóm cắt tỉa tối ưu18.
        
    - **Shortened LLAMA**: Giới thiệu cắt tỉa **độ sâu một lần**, chọn khối Transformer làm đơn vị cắt tỉa và đánh giá tầm quan trọng của chúng bằng **hàm lỗi và đạo hàm bậc hai của nó**19.
        
- **Magnitude-based Pruning**:
    
    - **FLAP**: Sử dụng một **metric biến động có cấu trúc** để đánh giá và xác định các cột trong ma trận trọng số phù hợp cho việc cắt tỉa20. Nó sử dụng cơ chế bù độ lệch đường cơ sở để khôi phục hiệu suất mà **không cần tinh chỉnh**21.
        
    - **SliceGPT**: Tận dụng tính bất biến tính toán của mạng transformer và tối ưu hóa quá trình cắt tỉa thông qua **Phân tích Thành phần Chính (PCA)** làm metric cắt tỉa22.
        
- **Regularization-based Pruning**:
    
    - **Sheared LLAMA**: Sử dụng **cặp nhân tử Lagrange** dựa trên mặt nạ cắt tỉa để áp đặt các ràng buộc trực tiếp lên hình dạng mô hình đã cắt tỉa, định hình cắt tỉa thành một bài toán tối ưu hóa có ràng buộc23.
        

---

### 4.3 Semi-Structured Pruning (Cắt tỉa bán cấu trúc)

**Ý chính dẫn dắt:**

Phương pháp này nằm giữa cắt tỉa có cấu trúc và không cấu trúc 24, cắt tỉa **một phần tham số** dựa trên các mẫu cụ thể (pattern) 25, có khả năng đạt được **cắt tỉa hạt mịn và điều chuẩn cấu trúc đồng thời**26. Độ thưa **N:M** (cứ M phần tử liên tiếp thì N phần tử không bằng không) là một ví dụ điển hình27.

**Các ví dụ đại diện:**

- **E-Sparse**: Triển khai độ thưa N:M bằng cách giới thiệu **entropy thông tin** làm metric để đánh giá tầm quan trọng của tham số28.
    
- **SparseGPT** và **Wanda**: Cả hai phương pháp này cũng **khám phá độ thưa N:M** cho LLMs29.
    
    - **SparseGPT** (trong N:M): Sử dụng phân vùng trọng số theo khối và **cắt tỉa N trọng số có lỗi tái tạo thấp nhất** (dựa trên thông tin Hessian) trong mỗi khối M30.
        
    - **Wanda** (trong N:M): Chia ma trận trọng số thành các nhóm M trọng số liên tiếp và **giữ lại N trọng số có điểm quan trọng cao nhất** (độ lớn x chuẩn kích hoạt) trong mỗi nhóm31313131.
        
- **Ampere Tensor Core GPU**: Hỗ trợ độ thưa bán cấu trúc hạt mịn **2:4** để tăng tốc Mạng nơ-ron thưa trên phần cứng này.

---

## 5. Knowledge Distillation (Chưng cất Tri thức - KD)

**Ý chính dẫn dắt:**

**Chưng cất Tri thức (KD)** 1là một kỹ thuật nhằm **chuyển giao tri thức** từ một mô hình lớn và phức tạp (gọi là **mô hình giáo viên** - teacher model) sang một mô hình nhỏ hơn và đơn giản hơn (gọi là **mô hình học sinh** - student model).

Các phương pháp KD được phân loại thành hai loại rõ ràng:

1. **Black-box KD** (KD Hộp đen): Chỉ có đầu ra của mô hình giáo viên là có thể truy cập, thường áp dụng cho LLMs nguồn đóng (closed-source).
    
2. **White-box KD** (KD Hộp trắng): Các tham số hoặc phân phối đầu ra của mô hình giáo viên đều có sẵn, thường áp dụng cho LLMs nguồn mở (open-source).
    

---

### 5.1 Black-box KD (Chưng cất Tri thức Hộp đen)

**Ý chính dẫn dắt:**

Black-box KD thường **sử dụng lời nhắc (prompt) để khiến LLM giáo viên tạo ra một tập dữ liệu chưng cất** (distillation dataset) để tinh chỉnh mô hình học sinh, từ đó chuyển giao năng lực6.

- **Mô hình Giáo viên**: Thường là LLMs nguồn đóng như **ChatGPT (gpt-3.5-turbo)** và **GPT4**.
    
- **Mô hình Học sinh**: Thường là các Mô hình Ngôn ngữ Nhỏ hơn (SLMs) như **GPT-2**, **T5**, **FlanT5**, và **CodeT5**.
    

Các phương pháp Black-box KD thường cố gắng **chưng cất các khả năng đột phá (emergent abilities)** của LLMs sang SLMs9:

#### 5.1.1 Chain-of-Thought (CoT) Distillation (Chưng cất Chuỗi suy nghĩ)

**Mô tả:** CoT yêu cầu LLMs **tạo ra các bước lý luận trung gian** (intermediate reasoning steps) để giải quyết các nhiệm vụ lý luận phức tạp từng bước một10.

**Các ví dụ đại diện:**

- **MT-COT**: Sử dụng LLMs để tạo ra lời giải thích và tận dụng **khung học tập đa nhiệm** (multi-task learning framework) để củng cố khả năng lý luận và khả năng tạo lời giải thích của các mô hình nhỏ hơn11.
    
- **Fine-tune-CoT**: Sử dụng kỹ thuật **Zero-shot CoT** để nhắc LLMs tạo ra **các lý do đa dạng** (diverse rationales) nhằm làm giàu tập dữ liệu chưng cất cho mô hình học sinh.
    
- **SOCRATIC COT**: Chưng cất **hai mô hình học sinh**: một bộ phân tách vấn đề (problem decomposer) và một bộ giải quyết vấn đề con (subproblem solver).
    
- **SCOTT**: Kết hợp **giải mã đối lập** (contrastive decoding) trong quá trình tạo lý do cho mô hình giáo viên và giải quyết các vấn đề lối tắt bằng cách đưa ra mục tiêu **lý luận phản thực tế** (counterfactual reasoning objective) trong quá trình huấn luyện mô hình học sinh14.
    
- **PaD**: Yêu cầu LLMs tạo ra lý do **Program-of-Thought (PoT)** thay vì CoT để xây dựng tập dữ liệu chưng cất, sau đó tinh chỉnh SLMs bằng tập dữ liệu này15.
    
- **DRA**: Giới thiệu cơ chế **học tập tự phản chiếu** (self-reflection learning mechanism), cho phép LLM học sinh học hỏi từ những sai lầm của chính nó và tăng cường khả năng lý luận16.
    
- **TDIG**: Phát hiện rằng **dữ liệu âm tính** (negative data) được tạo ra từ LLMs giáo viên cũng chứa tri thức lý luận, và hướng dẫn LLM học sinh học tri thức từ cả mẫu âm tính bên cạnh mẫu dương tính17.
    

#### 5.1.2 In-Context Learning (ICL) Distillation (Chưng cất Học tập trong Ngữ cảnh)

**Mô tả:** ICL sử dụng **các lời nhắc có cấu trúc với mô tả nhiệm vụ và các ví dụ** để LLMs học các nhiệm vụ mới mà không cần cập nhật gradient.

**Các ví dụ đại diện:**

- **In-context Learning Distillation (AICD)**: Chuyển giao khả năng học tập trong ngữ cảnh từ LLMs sang các mô hình nhỏ hơn bằng cách **kết hợp mục tiêu học tập trong ngữ cảnh với mục tiêu mô hình ngôn ngữ**. Nó bao gồm hai mô hình: **Meta In-context Tuning (Meta-ICT)**, và **Multitask In-context Tuning (Multitask-ICT)**20.
    
- **AICD (Learning to reason with autoregressive in-context distillation)**: Tận dụng tính chất tự hồi quy của LLMs để thực hiện **meta-teacher forcing** trên CoTs trong ngữ cảnh, cùng tối ưu hóa khả năng xảy ra của tất cả các CoTs trong ngữ cảnh, từ đó chưng cất khả năng học tập trong ngữ cảnh và lý luận vào các mô hình nhỏ hơn21.
    

#### 5.1.3 Instruction Following (IF) Distillation (Chưng cất Tuân thủ Hướng dẫn)

**Mô tả:** IF nhằm mục đích củng cố **khả năng zero-shot** của LLMs thông qua tinh chỉnh bằng một bộ sưu tập các cặp nhắc nhở-phản hồi giống như hướng dẫn22.

**Các ví dụ đại diện:**

- **Lion**: Nhắc LLM giáo viên **xác định và tạo ra các hướng dẫn "khó"** (hard instructions), sau đó được sử dụng để tăng cường năng lực của mô hình học sinh23.
    
- **LaMini-LM**: Phát triển một bộ sưu tập mở rộng gồm **2.58 triệu hướng dẫn** và tinh chỉnh một loạt các mô hình đa dạng bằng cách sử dụng các hướng dẫn này24.
    
- **SELF-INSTRUCT**: Sử dụng chính **LLM học sinh làm giáo viên để tự tạo tập dữ liệu tuân thủ hướng dẫn** (instruction following dataset), và tinh chỉnh chính mô hình học sinh bằng tập dữ liệu đó25.
    
- **Selective Reflection-Tuning**: Tận dụng LLMs giáo viên để **phản chiếu và cải thiện dữ liệu hiện có**, trong khi LLMs học sinh đánh giá và **lựa chọn kết hợp** những cải tiến này, từ đó tăng chất lượng dữ liệu và tính tương thích với LLMs học sinh26.
    

#### Remark 5 về Black-Box Distillation

- Black-box KD sử dụng đầu ra của mô hình giáo viên làm giám sát, nhưng đầu ra này có thể **không bao phủ tất cả các kịch bản đầu vào có thể có**27.
    
- Do đó, việc tìm hiểu cách xử lý **khả năng tổng quát hóa** của mô hình học sinh trên dữ liệu chưa biết và cách **tăng tính đa dạng của dữ liệu** là một lĩnh vực cần điều tra thêm28.
    

---

### 5.2 White-box KD (Chưng cất Tri thức Hộp trắng)

**Ý chính dẫn dắt:**

White-box KD cho phép LLM học sinh có được **sự hiểu biết sâu sắc hơn về cấu trúc nội bộ và biểu diễn tri thức** của LLM giáo viên, thường mang lại sự cải thiện hiệu suất cấp cao hơn29.

**Các ví dụ đại diện:**

- **MINILLM**: Là công trình đầu tiên nghiên cứu chưng cất từ LLMs sinh nguồn mở. Nó sử dụng mục tiêu **phân kỳ Kullback-Leibler ngược** (reverse Kullback-Leibler divergence objective), phù hợp hơn cho KD trên các mô hình ngôn ngữ sinh, để ngăn mô hình học sinh đánh giá quá cao các vùng xác suất thấp của phân phối giáo viên30.
    
- **GKD**: Khám phá việc chưng cất từ các **mô hình tự hồi quy**. Nó huấn luyện mô hình học sinh bằng cách sử dụng các đầu ra tự tạo, kết hợp **phản hồi của giáo viên**, và cho phép linh hoạt trong việc sử dụng các hàm lỗi khác nhau khi mô hình học sinh không thể tái tạo hoàn toàn phân phối của giáo viên31.
    
- **TED**: Đề xuất phương pháp chưng cất **từng lớp nhận biết nhiệm vụ** (task-aware layer-wise distillation), thiết kế các bộ lọc nhận biết nhiệm vụ nhằm **căn chỉnh các biểu diễn ẩn** của mô hình giáo viên và mô hình học sinh tại mỗi lớp trung gian, để giảm khoảng cách tri thức giữa chúng32.
    

#### Remark 6 về White-Box Distillation

- Mặc dù White-box KD cho phép học sinh học tri thức sâu hơn, nhưng hiện tại, **LLMs nguồn mở hoạt động kém hơn** so với LLMs nguồn đóng, điều này **hạn chế sự cải thiện hiệu suất** của mô hình học sinh trong White-box KD33.
    
- Một giải pháp khả thi là **chưng cất tri thức từ LLMs nguồn đóng sang LLMs nguồn mở** thông qua Black-box KD, và sau đó sử dụng White-box KD để **chuyển tri thức từ LLMs nguồn mở sang LLMs học sinh**34.
    

#### Remark 7 về Cấu trúc Nội bộ

- White-box KD thường liên quan đến việc hiểu và sử dụng **cấu trúc nội bộ** của LLMs, chẳng hạn như kết nối giữa các lớp và cài đặt tham số35.
    
- Việc khám phá sâu hơn về **các cấu trúc mạng khác nhau** và **sự tương tác giữa các lớp** có thể cải thiện hiệu quả của White-box KD.

---

## 6. Low-Rank Factorization (Phân tích Thừa số Hạng Thấp)

**Ý chính dẫn dắt:**

**Phân tích Thừa số Hạng Thấp** (Low-Rank Factorization) là kỹ thuật nhằm **giảm một ma trận lớn** (ví dụ: ma trận trọng số $W$) thành **các ma trận nhỏ hơn** (ví dụ: $U$ và $V$ để **tiết kiệm không gian và nỗ lực tính toán**.

Về mặt toán học, nó phân tích ma trận trọng số $W$ có kích thước $m \times n$ thành hai ma trận $U (m \times k)$  và $V$ ($k \times n$), trong đó **hạng $k$ nhỏ hơn nhiều** so với $m$ và $n$ ($W \approx UV$).

**Các ví dụ đại diện:**

- **LPLR (Low-Rank and Low-Precision Factorization)**: Nén ma trận trọng số của LLMs thông qua **phân tích thừa số hạng thấp ngẫu nhiên và độ chính xác thấp**.
    
    - Cụ thể, LPLR **xấp xỉ không gian cột** của ma trận bằng kỹ thuật **phác thảo ngẫu nhiên** (random sketching), lượng tử hóa các cột này, và sau đó chiếu các cột gốc lên không gian đã lượng tử hóa này để tạo ra hai thừa số hạng thấp được lưu trữ ở **độ chính xác thấp**.
        
- **ASVD (Activation-Aware Singular Value Decomposition)**: Phương pháp này giải quyết vấn đề **phân phối kích hoạt** (activation distribution) có ảnh hưởng đến hiệu suất nén.
    
    - ASVD đề xuất **chia tỷ lệ ma trận trọng số** bằng một ma trận đường chéo chứa các hệ số chia tỷ lệ tương ứng với **phân phối kích hoạt** của các kênh tính năng đầu vào6.
        
    - Hơn nữa, nó gán **tỷ lệ nén phù hợp nhất cho các lớp khác nhau** bằng cách phân tích **phân phối giá trị số ít** (singular values distribution) trong ma trận trọng số của từng lớp, đảm bảo tổn thất hiệu suất tối thiểu7.
        
- **LASER (Layer-Selective Rank Reduction)**: Chứng minh rằng hiệu suất của LLMs có thể được cải thiện đáng kể bằng cách áp dụng **Giảm hạng Chọn lọc Lớp** cho các lớp cụ thể của mô hình Transformer8.
    
    - LASER bao gồm việc **giảm hạng có chọn lọc** các thành phần bậc cao hơn của ma trận trọng số. Điều này được chứng minh là giúp cải thiện khả năng xử lý **dữ liệu huấn luyện hiếm** và khả năng **chống lại việc diễn giải lại câu hỏi** (question paraphrasing) của mô hình9.

---

## 7. Challenges and Future Directions (Thách thức và Định hướng Tương lai)

**Ý chính dẫn dắt:**

Nghiên cứu về các kỹ thuật nén mô hình cho Mô hình Ngôn ngữ Lớn (LLMs) vẫn đang ở giai đoạn đầu11. Các LLMs được nén hiện tại vẫn cho thấy một **khoảng cách hiệu suất đáng kể** so với các mô hình không nén2. Do đó, lĩnh vực này có nhiều thách thức cần giải quyết và các hướng nghiên cứu đầy hứa hẹn.

---

### 7.1 More Advanced Methods (Các Phương pháp Tiên tiến hơn)

- Nghiên cứu về các phương pháp nén mô hình **được điều chỉnh cụ thể** cho LLMs vẫn còn thiếu sót333.
    
- Bằng cách tìm hiểu sâu hơn về các phương pháp nén mô hình tiên tiến hơn, các nhà nghiên cứu có tiềm năng **tăng cường hiệu suất** của các LLMs được nén này4.
    

---

### 7.2 Scaling up Model Compression Methods from Other Models (Mở rộng các Phương pháp Nén từ các Mô hình Khác)

- Nhiều phương pháp nén mô hình kinh điển vẫn còn phổ biến trong các mô hình nhỏ truyền thống nhưng chưa được áp dụng rộng rãi cho LLMs5.
    
- Các phương pháp như **lottery tickets** (vé số) 6và **parameter sharing** (chia sẻ tham số) 7vẫn giữ tiềm năng đáng kể trong kỷ nguyên LLMs8.
    
- Công việc trong tương lai nên tập trung vào việc **mở rộng các phương pháp nén này sang LLMs** để đạt được khả năng nén sâu hơn9.
    

---

### 7.3 LLM Inference and Deployment (Suy luận và Triển khai LLM)

- **Hiệu quả khi triển khai** các LLMs được nén là một lĩnh vực quan trọng cần khám phá10.
    
- Điều này liên quan đến nhiều metric đánh giá, bao gồm **cường độ số học** (arithmetic intensity), **kích thước bộ nhớ** (memory size), và **thông lượng** (throughput)11.
    
- Có thể sử dụng công cụ phân tích như **Mô hình Roofline** (Roofline Model) để đánh giá **hiệu quả tài nguyên** của các LLMs được nén trên phần cứng cụ thể12.
    
- Việc đánh giá hiệu quả triển khai giúp hướng dẫn các nhà nghiên cứu trong việc **lựa chọn, phân tích ưu nhược điểm** của các phương pháp nén và tối ưu hóa chúng hơn nữa13.
    

---

### 7.4 The Effect of Scaling Law (Ảnh hưởng của Quy luật Mở rộng)

- **Quy luật Mở rộng** (Scaling Law) nhấn mạnh tác động đáng kể của kích thước mô hình, kích thước tập dữ liệu và tài nguyên tính toán đối với hiệu suất của LLMs14.
    
- Quy luật này đặt ra một **thách thức cơ bản** cho nén LLM: tồn tại sự **đánh đổi giữa kích thước mô hình và hiệu suất** trong các LLMs được nén15.
    
- Việc tìm hiểu sâu vào các cơ chế và lý thuyết làm nền tảng cho quy luật mở rộng là rất quan trọng để **làm sáng tỏ và có khả năng vượt qua giới hạn** này16.
    

---

### 7.5 AutoML for LLM Compression (AutoML cho Nén LLM)

- Các kỹ thuật nén hiện tại vẫn **phụ thuộc nhiều vào thiết kế thủ công**17. Ví dụ, việc thiết kế kiến trúc học sinh phù hợp cho chưng cất tri thức đòi hỏi nhiều nỗ lực của con người18.
    
- Một giải pháp khả thi là **kết hợp các kỹ thuật Automated Machine Learning (AutoML)** như **Meta-Learning** 19và **Neural Architecture Search (NAS)** 20với nén mô hình21.
    
- AutoML có thể **tự động chọn các siêu tham số** (hyperparameters) thích hợp và **điều chỉnh kiến trúc và quy mô** của các mô hình nén 22, từ đó giảm thiểu sự can thiệp của con người và chi phí liên quan23.
    
- Ngoài ra, AutoML có thể xác định **chiến lược nén tối ưu** phù hợp với yêu cầu nhiệm vụ cụ thể, tăng cường tỷ lệ nén mà không làm ảnh hưởng đến hiệu suất24.
    

---

### 7.6 Explainability of LLM Compression (Khả năng Giải thích của Nén LLM)

- Các nghiên cứu trước đây đã nêu ra mối lo ngại đáng kể về **khả năng giải thích** của các kỹ thuật nén được áp dụng cho Mô hình Ngôn ngữ Tiền huấn luyện (PLMs) 25, và những thách thức tương tự cũng mở rộng sang nén LLM26.
    
- Ví dụ, **CoT-distillation** có thể tăng cường hiệu suất lý luận của SLMs, nhưng **cơ chế** mà nó truyền tải khả năng CoT vẫn chưa rõ ràng27.
    
- Thách thức này nhấn mạnh tầm quan trọng của việc **tích hợp khả năng giải thích** (explainability) với các phương pháp nén mô hình28.
    
- Khả năng giải thích không chỉ làm rõ các thay đổi và sự đánh đổi trong quá trình nén mà còn **tăng cường hiệu quả và độ chính xác**29. Nó cũng hỗ trợ đánh giá hiệu suất của mô hình nén để đảm bảo nó phù hợp với các yêu cầu thực tế30.