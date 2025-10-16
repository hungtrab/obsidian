

# **Phân tích Tương tác Bậc cao trong Học tăng cường Giải thích được: Một Lộ trình Nghiên cứu**

## **Phần 1: Giả định về Tính cộng trong các Mô hình Giải thích Post-Hoc cho Học tăng cường**

Phần này sẽ xác thực một cách chặt chẽ giả thuyết cốt lõi rằng các phương pháp Giải thích Trí tuệ nhân tạo (XAI) hiện tại về cơ bản mang tính cộng. Phân tích sẽ đi sâu vào nền tảng toán học của các phương pháp XAI phổ biến để chứng minh rằng tính cộng là một lựa chọn thiết kế cơ bản, và lựa chọn này trở thành một điểm yếu nghiêm trọng khi đối mặt với các tương tác đặc trưng phức tạp.

### **1.1. Nền tảng Toán học của các Giải thích Cục bộ, Cộng tính**

Để hiểu được những hạn chế của các phương pháp giải thích hiện tại, điều cần thiết là phải phân tích cấu trúc toán học của chúng. Các phương pháp phổ biến nhất, chẳng hạn như LIME và SHAP, được xây dựng dựa trên một khuôn khổ vốn có tính cộng, nhằm đơn giản hóa các mô hình phức tạp thành các thành phần có thể diễn giải được.

Phân tích SHAP và LIME:  
SHAP (SHapley Additive exPlanations) và LIME (Local Interpretable Model-agnostic Explanations) là hai trong số các kỹ thuật XAI bất khả tri mô hình (model-agnostic) được sử dụng rộng rãi nhất.1 Nền tảng của chúng nằm ở việc xấp xỉ hành vi của một mô hình "hộp đen" phức tạp bằng một mô hình đơn giản hơn, có thể diễn giải được ở quy mô cục bộ.  
Cốt lõi của SHAP là giá trị Shapley từ lý thuyết trò chơi hợp tác, phân bổ "phần thưởng" (đầu ra của mô hình) một cách công bằng cho các "người chơi" (các đặc trưng đầu vào).3 Mô hình giải thích của SHAP được định nghĩa một cách rõ ràng là một phương pháp quy kết đặc trưng cộng tính.3 Công thức toán học của nó là:

g(z′)=ϕ0​+j=1∑M​ϕj​zj′​

Trong đó, g(z′) là mô hình giải thích, z′ là một vector liên minh biểu thị sự hiện diện (1) hoặc vắng mặt (0) của các đặc trưng, M là số lượng đặc trưng tối đa, và ϕj​ là giá trị Shapley (sự đóng góp) của đặc trưng thứ j. Công thức này, về bản chất, là một mô hình tuyến tính, nơi sự đóng góp của mỗi đặc trưng được cộng lại để tạo thành dự đoán cuối cùng.3  
Tương tự, LIME hoạt động bằng cách xây dựng một mô hình thay thế (surrogate model) tuyến tính, thưa thớt (sparse) xung quanh một dự đoán cụ thể để giải thích cách mô hình hộp đen hoạt động trong vùng lân cận đó.4 Bằng cách tạo ra các nhiễu loạn (perturbations) của một thực thể dữ liệu và quan sát các dự đoán tương ứng của mô hình hộp đen, LIME huấn luyện một mô hình tuyến tính có thể diễn giải để xấp xỉ hành vi cục bộ của mô hình phức tạp. Cấu trúc tuyến tính này vốn dĩ mang tính cộng, thể hiện dự đoán như là một tổng có trọng số của các đặc trưng.

Các Phương pháp dựa trên Gradient và Xấp xỉ Tuyến tính:  
Đối với các mô hình học tăng cường (RL) sử dụng đầu vào là hình ảnh (ví dụ: các tác tử chơi game Atari), các phương pháp giải thích phổ biến bao gồm bản đồ độ nổi bật (saliency maps) dựa trên gradient.6 Các phương pháp này, chẳng hạn như Vanilla Gradient, hoạt động bằng cách tính toán gradient của điểm số đầu ra (ví dụ: điểm số cho một hành động cụ thể) đối với các pixel đầu vào.7 Về mặt toán học, điều này dựa trên khai triển Taylor bậc nhất để xấp xỉ hàm điểm số phi tuyến tính cao  
Sc​(x) của mạng nơ-ron thành một hàm tuyến tính cục bộ 7:

Sc​(x)≈wTx+b

Trong đó, vector trọng số w chính là gradient: w=δxδSc​​∣x0​​. Ở đây, "sự giải thích" cho mỗi pixel đầu vào chính là giá trị của gradient tương ứng trong w. Điều này một lần nữa cho thấy một mô hình cộng tính, nơi sự thay đổi trong dự đoán được xấp xỉ bằng một tổng tuyến tính của những thay đổi trong các pixel đầu vào.  
Thuộc tính "Hoàn chỉnh" (Completeness):  
Một số phương pháp quy kết đường đi (path-attribution methods) như SHAP và Tích phân Gradient (Integrated Gradients) có thuộc tính "hoàn chỉnh".7 Điều này có nghĩa là tổng của tất cả các điểm số liên quan (relevance scores) của các đặc trưng đầu vào bằng chính xác sự khác biệt giữa dự đoán của mô hình cho đầu vào thực tế và dự đoán cho một đầu vào cơ sở (baseline).7 Mặc dù thuộc tính này đảm bảo rằng sự giải thích chiếm trọn vẹn sự thay đổi trong đầu ra, nó vẫn củng cố mô hình cộng tính: tổng hiệu ứng được phân chia và phân bổ một cách cộng tính cho các đặc trưng, thay vì mô hình hóa cách chúng tương tác để tạo ra hiệu ứng đó.  
Bản chất cộng tính của các phương pháp này không phải là một sai sót ngẫu nhiên mà là một lựa chọn thiết kế có chủ đích. Mục tiêu của XAI là tạo ra các giải thích đủ đơn giản để con người có thể hiểu được.8 Các mô hình tuyến tính và cộng tính là dạng giải thích dễ hiểu nhất. Tuy nhiên, sự đơn giản này đi kèm với một sự đánh đổi đáng kể về độ trung thực (faithfulness), đặc biệt là trong các miền phức tạp nơi các tương tác phi tuyến là yếu tố chi phối. Sự đánh đổi này trở nên đặc biệt nghiêm trọng trong RL, nơi các chiến lược tối ưu thường xuất phát từ sự kết hợp phức tạp của các yếu tố trạng thái, chứ không phải từ tổng các phần riêng lẻ của chúng.

### **1.2. Hạn chế Cố hữu: Cạm bẫy của Sự độc lập Đặc trưng và Đa cộng tuyến**

Các mô hình giải thích cộng tính dựa trên một số giả định cơ bản về dữ liệu, và khi các giả định này bị vi phạm—điều thường xuyên xảy ra trong các môi trường RL phức tạp—độ tin cậy của chúng sẽ bị suy giảm nghiêm trọng.

Giả định về Sự độc lập Đặc trưng:  
Một trong những giả định quan trọng nhất, và cũng thường bị vi phạm nhất, của các phương pháp như KernelSHAP là các đặc trưng độc lập với nhau.1 Khi mô phỏng sự "vắng mặt" của một đặc trưng để tính toán giá trị Shapley, phương pháp này lấy giá trị trung bình trên phân phối biên của dữ liệu.3 Quá trình này chỉ có ý nghĩa về mặt thống kê nếu các đặc trưng không tương quan. Tuy nhiên, trong nhiều kịch bản thực tế, các đặc trưng có mối tương quan cao. Ví dụ, trong một môi trường RL, vị trí của tác tử và khoảng cách của nó đến kẻ thù gần nhất là hai đặc trưng có tương quan chặt chẽ. Việc mô phỏng sự vắng mặt của "vị trí" trong khi vẫn giữ lại "khoảng cách đến kẻ thù" sẽ tạo ra một trạng thái dữ liệu phi thực tế, không thể xảy ra, dẫn đến các giá trị Shapley không đáng tin cậy.1  
Thất bại trong việc Nắm bắt các Tương tác Phi tuyến:  
LIME, với thiết kế sử dụng một mô hình thay thế tuyến tính cục bộ, vốn dĩ không thể nắm bắt được các phụ thuộc phi tuyến giữa các đặc trưng.1 Nếu giá trị của một hành động phụ thuộc vào tích của hai đặc trưng trạng thái (ví dụ:  
giá\_trị \= w \* đặc\_trưng\_A \* đặc\_trưng\_B), một mô hình tuyến tính sẽ không bao giờ có thể xấp xỉ chính xác mối quan hệ này. SHAP được ghi nhận là có *tiềm năng* lớn hơn trong việc phát hiện các liên kết phi tuyến, vì nó xem xét các liên minh đặc trưng khác nhau. Tuy nhiên, đầu ra cuối cùng của nó vẫn là một sự quy kết cộng tính của các hiệu ứng này, chứ không phải là một mô hình tường minh về chính sự tương tác đó.1 Nó có thể cho chúng ta biết rằng cả

đặc\_trưng\_A và đặc\_trưng\_B đều quan trọng, nhưng nó không thể hiện rõ ràng rằng tầm quan trọng của chúng đến từ sự kết hợp của cả hai.

Sự bất ổn và Thiếu nhất quán:  
Do quá trình lấy mẫu ngẫu nhiên trong việc tạo ra các nhiễu loạn, các giải thích của LIME có thể không ổn định và thay đổi giữa các lần chạy khác nhau trên cùng một thực thể.10 Ngược lại, SHAP có các đảm bảo lý thuyết về tính nhất quán (consistency), nghĩa là nếu một mô hình thay đổi theo cách làm tăng tác động thực tế của một đặc trưng, thì giá trị Shapley của nó sẽ không giảm.3 Tuy nhiên, ngay cả với sự đảm bảo này, các kết quả của SHAP vẫn có thể gây hiểu lầm khi các giả định cơ bản của nó (như sự độc lập đặc trưng) bị phá vỡ bởi đa cộng tuyến.1 Điều này cho thấy rằng ngay cả với các đảm bảo lý thuyết, việc áp dụng thực tế các mô hình cộng tính này vẫn tiềm ẩn nhiều rủi ro trong các miền phức tạp.  
Tóm lại, giả định về tính cộng trong các phương pháp XAI post-hoc phổ biến là một con dao hai lưỡi. Nó mang lại sự đơn giản và dễ hiểu, nhưng lại phải trả giá bằng việc không thể nắm bắt trung thực các tương tác đặc trưng phức tạp, vốn là cốt lõi của việc ra quyết định chiến lược trong nhiều nhiệm vụ học tăng cường.

## **Phần 2: Vượt ra ngoài Tính cộng: Mô hình hóa Tương tác Đặc trưng Bậc cao**

Phần này sẽ thiết lập tính hợp lệ lý thuyết của giải pháp được đề xuất và kết nối nó với các nghiên cứu tiên tiến trong lĩnh vực XAI. Phân tích sẽ cho thấy rằng lĩnh vực này đang tích cực hướng tới việc nắm bắt các tương tác đặc trưng, tạo ra một nền tảng vững chắc cho công việc nghiên cứu được đề xuất.

### **2.1. Mô hình hóa Tương tác Tường minh bằng các Thành phần Đa thức và Bậc cao**

Ý tưởng sử dụng các mô hình bậc cao để nắm bắt các tương tác phi tuyến không phải là mới; nó là một kỹ thuật đã được thiết lập trong học máy và thống kê. Việc chính thức hóa ý tưởng này cung cấp một phương pháp cụ thể để triển khai và kiểm tra giả thuyết nghiên cứu.

Chính thức hóa Mô hình Bậc cao:  
Mô hình được đề xuất, có dạng f(x)=x1​w1​+x2​w2​+x1​x2​w12​, là một ví dụ kinh điển của một mô hình đa thức bậc hai. Thành phần x1​x2​w12​ được gọi là thành phần tương tác (interaction term). Nó cho phép mô hình nắm bắt được hiệu ứng hiệp đồng (synergistic effect) hoặc đối kháng (antagonistic effect) giữa các đặc trưng x1​ và x2​. Nếu không có thành phần này, mô hình sẽ chỉ có thể nắm bắt được các đóng góp cộng tính riêng lẻ của từng đặc trưng.12  
Kết nối với PolynomialFeatures:  
Trong thực tế, việc tạo ra các đặc trưng tương tác này có thể được tự động hóa bằng cách sử dụng các thư viện học máy tiêu chuẩn. Ví dụ, lớp PolynomialFeatures trong thư viện Scikit-learn của Python có thể biến đổi một tập dữ liệu đầu vào bằng cách tạo ra tất cả các kết hợp đa thức của các đặc trưng cho đến một bậc nhất định.14 Ví dụ, với đầu vào là  
\[x1​,x2​\] và bậc là 2, phép biến đổi này sẽ tạo ra một tập đặc trưng mới: \[1,x1​,x2​,x12​,x22​,x1​x2​\]. Sau đó, một mô hình hồi quy tuyến tính đơn giản được huấn luyện trên tập đặc trưng mới này có thể học được các trọng số tương ứng, bao gồm cả trọng số w12​ cho thành phần tương tác x1​x2​.12 Điều này cung cấp một phương pháp cụ thể và thực tiễn để triển khai ý tưởng cốt lõi của nghiên cứu.

Mạng Nơ-ron Đa thức (PNNs):  
Ở một mức độ phức tạp hơn, khái niệm này có thể được mở rộng thành các kiến trúc mô hình tiên tiến hơn. Mạng Nơ-ron Đa thức (Polynomial Neural Networks \- PNNs) là một loại kiến trúc mạng vốn có cấu trúc để mô hình hóa các tương tác này một cách tự nhiên.15 Các nghiên cứu gần đây đã khám phá các thuộc tính về khả năng nhận dạng (identifiability) và cấu trúc của chúng, cho thấy rằng ý tưởng về mô hình hóa đa thức có thể mở rộng sang các mô hình phức tạp hơn, phù hợp với các tác tử học sâu.15

### **2.2. Sự trỗi dậy của các Khuôn khổ Giải thích Bậc cao và Nhân tính**

Nhận thức được những hạn chế của các phương pháp cộng tính, cộng đồng nghiên cứu XAI đã và đang phát triển các khuôn khổ tiên tiến hơn để nắm bắt một cách tường minh các tương tác đặc trưng.

Các Chỉ số Tương tác Shapley:  
Một hướng nghiên cứu quan trọng là mở rộng giá trị Shapley để đo lường các tương tác. Thay vì chỉ tính một giá trị đóng góp ϕi​ cho mỗi đặc trưng, các phương pháp này tính toán các giá trị cho các cặp, bộ ba đặc trưng, v.v. Các chỉ số này bao gồm:

* **Chỉ số Tương tác Shapley (SII \- Shapley Interaction Index):** Đo lường hiệu ứng tương tác thuần túy giữa hai đặc trưng.16  
* **Chỉ số Tương tác Taylor Shapley (STI \- Shapley Taylor Interaction Index):** Cung cấp một cách phân rã khác cho các tương tác.16  
* **Chỉ số Tương tác Shapley Trung thực (FSI \- Faithful Shapley Interaction Index):** Một biến thể khác nhằm cải thiện các thuộc tính của các chỉ số trước đó.16

Các phương pháp này, được gọi chung là Tương tác Shapley (Shapley Interactions \- SIs), cho phép phân tích sâu hơn về cách các nhóm đặc trưng hợp tác hoặc cản trở nhau để tạo ra một dự đoán, trực tiếp giải quyết vấn đề cốt lõi của tính cộng.17

Các Công cụ Thực tiễn: TreeSHAP-IQ và shapiq:  
Những khái niệm này không chỉ dừng lại ở lý thuyết. Các công cụ phần mềm đã được phát triển để biến chúng thành hiện thực. Thư viện Python shapiq cung cấp các triển khai cho việc xấp xỉ các Tương tác Shapley bậc bất kỳ cho các mô hình chung.17 Quan trọng hơn, TreeSHAP-IQ là một thuật toán hiệu quả được thiết kế đặc biệt để tính toán các tương tác này cho các mô hình dựa trên cây (ví dụ: XGBoost, LightGBM), vốn rất phổ biến trong nhiều ứng dụng học máy.16 Sự tồn tại của các công cụ này cung cấp một con đường rõ ràng để áp dụng các phương pháp giải thích tiên tiến này trong nghiên cứu của bạn.  
SHAP Nhân tính (mSHAP):  
Một mô hình thay thế khác, đặc biệt phù hợp cho các lĩnh vực mà các hiệu ứng có bản chất nhân tính thay vì cộng tính (ví dụ: mô hình hóa rủi ro trong bảo hiểm), là SHAP Nhân tính (Multiplicative SHAP \- mSHAP).19 Phương pháp này hoạt động bằng cách biến đổi vấn đề sang không gian logarit, nơi phép nhân trở thành phép cộng. Sau đó, SHAP tiêu chuẩn được áp dụng trong không gian logarit này, và kết quả cuối cùng được biến đổi ngược lại bằng hàm mũ.19 Cách tiếp cận này duy trì các thuộc tính mong muốn như "hiệu quả nhân tính" (multiplicative efficiency) và đã được triển khai trong các gói phần mềm có sẵn như  
mshap trong R.21

Sự tồn tại của các khuôn khổ như TreeSHAP-IQ và mSHAP cho thấy cộng đồng XAI đã nhận thức rõ những hạn chế của SHAP cộng tính. Điều này định vị một cách chiến lược khoảng trống nghiên cứu. Cơ hội nghiên cứu thực sự không phải là phát minh ra một phương pháp giải thích bậc cao hoàn toàn mới từ đầu, mà là trở thành người đầu tiên *áp dụng một cách có hệ thống và đánh giá một cách chặt chẽ* các kỹ thuật mới nổi này trong bối cảnh độc đáo của Học tăng cường. Trong RL, các tương tác đặc trưng không chỉ là một vấn đề thống kê cần khắc phục; chúng là bản chất của việc ra quyết định chiến lược. Việc thu hẹp khoảng cách này—bằng cách điều chỉnh, áp dụng và tạo ra các thước đo đánh giá mới cho các phương pháp XAI tiên tiến này dành riêng cho miền XRL—sẽ là một đóng góp độc đáo và có giá trị.

## **Phần 3: Khảo sát các Môi trường Học tăng cường Thể hiện Thuộc tính Nhân tính**

Phần này xác định và phân tích một danh mục các môi trường RL có thể đóng vai trò là nơi thử nghiệm. Tiêu chí chính là sự hiện diện của các tương tác đặc trưng phi tuyến, mạnh mẽ mà một tác tử bắt buộc phải nắm vững để hoàn thành nhiệm vụ.

### **3.1. Định nghĩa Thuộc tính "AND" trong Học tăng cường**

Để tiến hành một nghiên cứu có hệ thống, cần phải chính thức hóa khái niệm "thuộc tính nhân" hay "thuộc tính AND" trong khuôn khổ của một Quá trình Quyết định Markov (MDP).23

Giả sử Vπ(s) là hàm giá trị của một trạng thái s dưới một chính sách π. Một trạng thái có thể được biểu diễn bằng một tập hợp các đặc trưng. Xét ba trạng thái:

* sA​: một trạng thái chứa đặc trưng A nhưng không chứa đặc trưng B.  
* sB​: một trạng thái chứa đặc trưng B nhưng không chứa đặc trưng A.  
* sAB​: một trạng thái chứa cả hai đặc trưng A và B.

Thuộc tính "AND" được thể hiện nếu bất đẳng thức sau đây được thỏa mãn:

Vπ(sAB​)≫Vπ(sA​)+Vπ(sB​)  
Bất đẳng thức này biểu thị một hiệu ứng hiệp đồng (synergy). Giá trị của việc có cả hai đặc trưng cùng lúc lớn hơn đáng kể so với tổng giá trị của việc có từng đặc trưng riêng lẻ. Điều này chính thức hóa ví dụ của giáo sư: giá trị của việc gặp cả cờ đỏ và cờ xanh cùng lúc không đơn giản là tổng giá trị của việc gặp từng cờ riêng lẻ; nó tạo ra một tình huống chiến lược mới với một giá trị riêng biệt.

### **3.2. Phân tích các Môi trường Chế tạo và Xây dựng**

Các môi trường sandbox nơi việc chế tạo (crafting) và xây dựng là trọng tâm cung cấp những ví dụ rõ ràng và trực quan nhất về thuộc tính "AND".

Minecraft:  
Minecraft là một ví dụ kinh điển. Để chế tạo một chiếc cuốc gỗ, tác tử phải có cả gỗ (logs) và một bàn chế tạo (crafting table) trong kho đồ của mình. Sự hiện diện của một trong hai mà không có cái còn lại là vô dụng cho mục đích này. Đây là một cổng logic "AND" rời rạc và hoàn hảo.24 Cây công nghệ (tech tree) phức tạp của trò chơi, từ việc chế tạo các công cụ cơ bản bằng gỗ đến việc nấu chảy quặng sắt và chế tạo các vật phẩm cao cấp hơn, tạo ra một hệ thống phân cấp của các tương tác "AND" như vậy.25 Các khuôn khổ mã nguồn mở như Malmo 27, MineRL 24, và Voyager 30 cung cấp các API cần thiết để xây dựng và thử nghiệm các tác tử trong môi trường phong phú này.  
Factorio:  
Môi trường này nâng khái niệm tương tác lên quy mô công nghiệp. Để sản xuất một "mạch điện tử", tác tử phải thiết lập cả một dây chuyền sản xuất tấm sắt và một dây chuyền sản xuất dây đồng, cả hai cùng cung cấp nguyên liệu cho một máy lắp ráp. Bất kỳ sự gián đoạn nào trong một trong hai dòng đầu vào sẽ làm cho toàn bộ dây chuyền sản xuất trở nên vô dụng. Môi trường Học tập Factorio (Factorio Learning Environment \- FLE) cung cấp một nền tảng tiêu chuẩn để đánh giá loại hình lập kế hoạch dài hạn và lý luận tổ hợp phức tạp này.31  
Các Trò chơi Sinh tồn/Sandbox khác:  
Để chứng minh tính tổng quát của loại môi trường này, có thể kể đến Don't Starve 34 và  
*Terraria*.36 Cả hai đều có các cơ chế chế tạo và sinh tồn phức tạp, đòi hỏi người chơi phải kết hợp các tài nguyên theo những công thức cụ thể để tạo ra các công cụ, vũ khí và cấu trúc cần thiết để sống sót.

### **3.3. Phân tích các Môi trường Trò chơi Chiến lược và Tổ hợp**

Các trò chơi chiến lược thời gian thực (RTS) và các trò chơi giải đố cung cấp một loại tương tác khác, nơi thuộc tính "AND" thể hiện dưới dạng các thành phần chiến lược hoặc các cơ chế logic.

StarCraft II:  
Trong StarCraft II, thuộc tính "AND" biểu hiện dưới dạng các đội hình quân đội chiến lược. Ví dụ, một nhóm lính thủy đánh bộ (Marines) và một tàu cứu thương (Medivac) có giá trị chiến đấu cao hơn nhiều so với tổng giá trị của chúng khi hoạt động riêng lẻ, do khả năng hồi máu của Medivac làm tăng đáng kể sức chịu đựng của các đơn vị Marines mỏng manh. Tương tự, sự kết hợp giữa High Templar (gây sát thương lan) và Zealots (gây sát thương đơn mục tiêu cận chiến) tạo thành một đội hình cổ điển có thể đối phó với nhiều loại kẻ thù khác nhau. Không gian hành động tổ hợp khổng lồ của trò chơi đòi hỏi các tác tử phải suy luận về các sự kết hợp này để thành công.38 Môi trường PySC2 giúp cho việc nghiên cứu trong lĩnh vực này trở nên dễ dàng tiếp cận.39  
Xây dựng Tổ hợp và Giải đố:  
Một ví dụ khác là bài toán "xây dựng tổ hợp" (combinatorial construction), nơi một tác tử lắp ráp các khối giống như LEGO.40 Tính hợp lệ của việc đặt một khối mới phụ thuộc vào cấu hình kết hợp của  
*tất cả* các khối hiện có. Đây là một dạng tương tác mạnh mẽ và phức tạp. Ngoài ra, các trò chơi giải đố như *Portal* hoặc *The Talos Principle* 41 có thể truyền cảm hứng cho việc tạo ra một môi trường Gym tùy chỉnh.43 Trong môi trường như vậy, một tác tử có thể cần phải tạo ra một cổng màu xanh

*và* một cổng màu đỏ để giải một câu đố, hoặc phải nhấn một nút *và* đặt một khối lên một công tắc khác đồng thời để mở một cánh cửa.

### **3.4. Phân tích các Môi trường Phối hợp Đa tác tử**

Trong Học tăng cường Đa tác tử (MARL), thuộc tính "AND" có thể biểu hiện dưới dạng sự cần thiết của các hành động phối hợp, đồng thời.

Thư viện PettingZoo, đặc biệt là bộ Môi trường Đa hạt (Multi-Particle Environments \- MPE), cung cấp các ví dụ điển hình.44 Trong các nhiệm vụ hợp tác như "Simple Spread", thành công đòi hỏi tất cả các tác tử phải bao phủ tất cả các mốc cùng một lúc. Phần thưởng được tối đa hóa chỉ khi Tác tử 1 ở Mốc 1

*VÀ* Tác tử 2 ở Mốc 2, v.v..45 Đây là một biểu hiện của thuộc tính "AND" về không gian và thời gian, dựa trên sự phối hợp của các tác tử.

Để tổng hợp các phân tích trên, bảng sau đây cung cấp một so sánh có cấu trúc về các môi trường đã thảo luận, giúp đưa ra quyết định sáng suốt về việc lựa chọn môi trường phù hợp nhất cho các mục tiêu nghiên cứu cụ thể.

**Bảng 1: Phân tích So sánh các Môi trường RL để Nghiên cứu Tương tác Đặc trưng**

| Môi trường | Loại Thuộc tính 'AND' | Độ phức tạp Không gian Trạng thái/Hành động | Tính sẵn có của API Mã nguồn mở | Mức độ Phù hợp cho Nghiên cứu XRL |
| :---- | :---- | :---- | :---- | :---- |
| **Minecraft** | Chế tạo, Sinh tồn | Rất cao | Cao (MineRL, Malmo) | Xuất sắc |
| **Factorio** | Chuỗi Tự động hóa | Cực kỳ cao | Trung bình (FLE) | Xuất sắc |
| **StarCraft II** | Thành phần Chiến lược | Cực kỳ cao | Cao (PySC2) | Rất tốt |
| **PettingZoo MPE** | Phối hợp Tác tử | Thấp đến Trung bình | Cao (PettingZoo) | Tốt (cho sự phối hợp) |
| **Giải đố Tùy chỉnh** | Logic, Cơ chế | Thay đổi | Không có (cần phát triển) | Cao (được thiết kế riêng cho vấn đề) |

Các môi trường tốt nhất cho nghiên cứu này, như Minecraft và Factorio, không chỉ có các tương tác "AND" đơn lẻ. Chúng có một đồ thị phụ thuộc phân cấp sâu sắc của các tương tác (một "cây công nghệ"). Ví dụ, trong Minecraft, chuỗi phụ thuộc là: gỗ → ván gỗ → bàn chế tạo → cuốc gỗ → đá cuội → lò nung → quặng sắt → thỏi sắt. Cấu trúc này cung cấp một chương trình giảng dạy tự nhiên (natural curriculum). Một phương pháp XRL trước tiên có thể được kiểm tra về khả năng giải thích tương tác đơn giản gỗ → ván gỗ, và sau đó được mở rộng để giải thích chuỗi phức tạp hơn nhiều dẫn đến một thỏi sắt. Cấu trúc phân cấp này là nơi thử nghiệm lý tưởng để đánh giá khả năng mở rộng và chiều sâu của các mô hình giải thích bậc cao, vì nó cung cấp một cách có ý nghĩa về mặt ngữ nghĩa để kiểm tra và xác thực sự cần thiết của các giải thích có bậc ngày càng tăng.

## **Phần 4: Các Hướng Nghiên cứu Mới trong Học tăng cường Giải thích được**

Phần này chuyển từ phân tích sang tổng hợp, đề xuất ba hướng nghiên cứu cụ thể, có khả năng xuất bản. Chúng được trình bày theo một phổ tăng dần về tính mới lạ và độ khó kỹ thuật, cung cấp một lựa chọn chiến lược cho chương trình nghị sự nghiên cứu.

### **4.1. Hướng Nghiên cứu 1: Đánh giá Chuẩn các Mô hình XAI Bậc cao trong các Môi trường Giàu Tương tác**

Đây là một hướng đi thực nghiệm mạnh mẽ, tập trung vào việc chứng minh một cách định lượng sự vượt trội của các phương pháp giải thích bậc cao.

Mục tiêu:  
Chứng minh một cách thực nghiệm rằng các phương pháp XAI cộng tính hiện có thất bại trong các môi trường RL giàu tương tác, trong khi các phương pháp bậc cao mới nổi cung cấp các giải thích trung thực hơn.  
**Phương pháp luận:**

1. **Lựa chọn Môi trường:** Chọn một môi trường có hệ thống phân cấp tương tác rõ ràng, chẳng hạn như một nhiệm vụ chế tạo đơn giản trong Minecraft sử dụng các khuôn khổ Malmo hoặc MineRL.24  
2. **Huấn luyện Tác tử:** Huấn luyện một tác tử tiêu chuẩn như Deep Q-Network (DQN) hoặc Proximal Policy Optimization (PPO) bằng cách sử dụng một thư viện như Stable Baselines 3 47 để thành thạo nhiệm vụ. Mạng Q hoặc mạng Giá trị của tác tử sẽ đóng vai trò là mô hình cần được giải thích.  
3. **Áp dụng các Giải thích:** Đối với một tập hợp các trạng thái quan trọng (ví dụ: trạng thái có 2/3 nguyên liệu cần thiết cho một công thức), tạo ra các giải thích bằng cách sử dụng:  
   * *Phương pháp Cộng tính Cơ sở:* KernelSHAP được áp dụng cho mạng Q của tác tử.49  
   * *Phương pháp Bậc cao Đề xuất:* Một triển khai của TreeSHAP-IQ (nếu sử dụng tác tử dựa trên cây) hoặc một phương pháp xấp xỉ Tương tác Shapley bậc 2/3.16  
4. **Đánh giá (Đóng góp Cốt lõi):** Phát triển và áp dụng các thước đo độ trung thực (faithfulness metrics) được điều chỉnh cho RL. Phương pháp chính sẽ là một **kiểm tra dựa trên nhiễu loạn (perturbation-based test)**.52 Ý tưởng là nếu một giải thích là trung thực, việc loại bỏ các đặc trưng mà nó cho là quan trọng nhất sẽ gây ra sự sụt giảm lớn nhất trong hiệu suất của mô hình (ví dụ: giá trị Q của hành động tối ưu).  
   **Mã giả cho Kiểm tra Độ trung thực:**  
   Python  
   def faithfulness\_test(agent, state, explanation, k):  
       """  
       Đánh giá độ trung thực của một giải thích bằng cách gây nhiễu loạn.

       :param agent: Tác tử RL đã được huấn luyện.  
       :param state: Trạng thái hiện tại cần giải thích.  
       :param explanation: Các điểm quan trọng của đặc trưng từ một phương pháp XAI.  
       :param k: Số lượng đặc trưng quan trọng nhất cần gây nhiễu loạn.  
       :return: Điểm số độ trung thực (ví dụ: sự thay đổi trong giá trị Q).  
       """

       \# 1\. Lấy giá trị Q ban đầu  
       original\_q\_values \= agent.predict\_q\_values(state)  
       optimal\_action\_q\_value \= max(original\_q\_values)

       \# 2\. Xác định các đặc trưng quan trọng nhất theo giải thích  
       important\_feature\_indices \= explanation.get\_top\_k\_features(k)

       \# 3\. Gây nhiễu loạn trạng thái bằng cách loại bỏ/che giấu các đặc trưng quan trọng  
       \# Lưu ý: Phép nhiễu loạn phải thực tế và không tạo ra trạng thái OOD \[54\]  
       \# Ví dụ: loại bỏ một vật phẩm khỏi kho đồ, thay vì đặt các giá trị pixel thành 0\.  
       perturbed\_state \= state.mask\_features(important\_feature\_indices)

       \# 4\. Tính giá trị Q sau khi gây nhiễu loạn  
       perturbed\_q\_values \= agent.predict\_q\_values(perturbed\_state)  
       perturbed\_optimal\_action\_q\_value \= max(perturbed\_q\_values)

       \# 5\. Tính điểm số độ trung thực  
       \# Một giải thích trung thực sẽ dẫn đến sự sụt giảm lớn về giá trị Q  
       faithfulness\_score \= optimal\_action\_q\_value \- perturbed\_optimal\_action\_q\_value

       return faithfulness\_score

   Nghiên cứu sẽ so sánh các điểm số độ trung thực của các phương pháp cộng tính so với các phương pháp bậc cao. Giả thuyết là các phương pháp bậc cao sẽ vượt trội đáng kể. Điều này kết nối với các khuôn khổ đánh giá mạnh mẽ được đề xuất trong các công trình như XAI-TRIS và M4.55

### **4.2. Hướng Nghiên cứu 2: Giải thích Phản thực tế để Thăm dò Chiến lược của Tác tử**

Hướng đi này chuyển mô hình từ quy kết (attribution) sang phản thực tế (counterfactuals), đặt ra câu hỏi: "Sự thay đổi trạng thái tối thiểu nào là cần thiết để thay đổi hành động tối ưu của tác tử?"

Mục tiêu:  
Tạo ra các giải thích phản thực tế để tiết lộ sự hiểu biết của tác tử về các thuộc tính "AND" cần thiết để thành công.  
**Phương pháp luận:**

1. **Kịch bản:** Xét một trạng thái trong Minecraft nơi tác tử có gỗ và bàn chế tạo nhưng cần đá để chế tạo cuốc đá. Hành động tối ưu hiện tại của tác tử có thể là "khám phá để tìm đá". Một người dùng có thể hỏi, "Tại sao tác tử không chế tạo cuốc?"  
2. **Tạo ra Phản thực tế:** Mục tiêu là tìm ra nhiễu loạn tối thiểu đối với trạng thái s để tạo ra một trạng thái phản thực tế s′ nơi hành động tối ưu của tác tử π(s′) trở thành "chế tạo cuốc". Điều này có thể được định hình như một bài toán tối ưu hóa, như đã được thảo luận trong các tài liệu liên quan.57  
3. **Kiểm tra Thuộc tính "AND":** Một giải thích phản thực tế thành công sẽ xác định rằng việc thêm "đá" vào kho đồ là sự thay đổi tối thiểu cần thiết. Điều này chứng tỏ rằng chính sách của tác tử phụ thuộc vào sự *kết hợp* của gỗ, bàn chế tạo, *và* đá. Nó trực tiếp xác nhận sự hiểu biết của tác tử về quy tắc nhân tính này.

Thách thức:  
Việc tạo ra các giải thích phản thực tế trong không gian trạng thái có chiều cao (như hình ảnh) là rất khó và thường đòi hỏi các mô hình sinh (generative models).57 Tuy nhiên, đối với các môi trường RL mang tính biểu tượng như Minecraft, điều này khả thi hơn vì các nhiễu loạn liên quan đến việc thay đổi các vật phẩm rời rạc trong kho đồ.

### **4.3. Hướng Nghiên cứu 3: Từ Tương quan đến Nhân quả với XRL Nhân quả**

Đây là một hướng đi tham vọng hơn, mang tính lý thuyết, nhằm vượt ra ngoài các giải thích dựa trên tương quan để cung cấp các giải thích nhân quả.

Mục tiêu:  
Vượt ra ngoài các giải thích tương quan (như SHAP) và cung cấp các giải thích nhân quả cho chính sách của một tác tử. Mục tiêu là giải thích tại sao một sự kết hợp của các đặc trưng lại dẫn đến một trạng thái có giá trị cao.  
**Phương pháp luận:**

1. **Học Mô hình Nhân quả:** Cốt lõi của phương pháp này là học một Mô hình Nhân quả Cấu trúc (Structural Causal Model \- SCM) về động lực học của môi trường như được tác tử cảm nhận.60 SCM này sẽ biểu diễn các mối quan hệ như:  
   (có\_gỗ, có\_bàn\_chế\_tạo) \-\> có\_thể\_chế\_tạo\_ván\_gỗ.  
2. **Giải thích Nhân quả:** Sự giải thích cho một hành động (ví dụ: "di chuyển đến bàn chế tạo") sẽ không dựa trên sự quy kết đặc trưng mà dựa trên chuỗi nhân quả đã học được: "Tác tử đang di chuyển đến bàn chế tạo vì nó tin rằng hành động này sẽ *gây ra* việc trạng thái có\_thể\_chế\_tạo\_ván\_gỗ trở thành sự thật, và trạng thái này lại nằm trên con đường nhân quả dẫn đến mục tiêu có phần thưởng cao là chế tạo một chiếc cuốc kim cương."

Tính Mới lạ và Tham vọng:  
Đây là một hướng đi rất tham vọng, phù hợp với lĩnh vực mới nổi của Học tăng cường Nhân quả (Causal Reinforcement Learning).61 Một triển khai thành công sẽ đại diện cho một bước nhảy vọt cơ bản trong XRL, chuyển từ việc giải thích "cái gì" (đặc trưng nào quan trọng) sang "tại sao" chúng quan trọng theo một nghĩa cơ học. Nó sẽ trả lời các câu hỏi phản thực tế "tại sao không" với mức độ chặt chẽ cao hơn nhiều so với phương pháp trong Hướng đi 2.60  
Ba hướng đi này không loại trừ lẫn nhau mà tạo thành một lộ trình nghiên cứu logic. Hướng đi 1 (Đánh giá chuẩn) là một đóng góp thực nghiệm mạnh mẽ, rủi ro tương đối thấp và cần thiết để thiết lập tầm quan trọng của vấn đề trong RL. Hướng đi 2 (Phản thực tế) xây dựng trên đó bằng cách đề xuất một phương pháp giải thích trực quan và có mục tiêu hơn. Hướng đi 3 (Nhân quả) là mục tiêu lý thuyết cuối cùng, cung cấp hình thức hiểu biết sâu sắc nhất. Cấu trúc này cung cấp một kế hoạch nghiên cứu kéo dài nhiều năm, có khả năng bao gồm một luận văn Thạc sĩ và một luận án Tiến sĩ, cho phép lựa chọn điểm khởi đầu dựa trên mục tiêu, nguồn lực và mức độ chấp nhận rủi ro.

## **Phần 5: Tổng hợp và Khuyến nghị**

Phần cuối cùng này sẽ cung cấp một bản tóm tắt ngắn gọn các phát hiện của báo cáo và đưa ra một khuyến nghị rõ ràng, có thể hành động về con đường hứa hẹn nhất cho bài báo nghiên cứu ban đầu.

Xác nhận Giả thuyết:  
Phân tích kết luận rằng giả thuyết ban đầu của giáo sư không chỉ đúng mà còn chỉ ra một lĩnh vực nghiên cứu quan trọng và đang được tích cực khám phá trong XAI. Các phương pháp cộng tính hiện tại về cơ bản không phù hợp với các miền phức tạp, nơi các tương tác là động lực chính, và đây chính là nơi các tác tử RL hiện đại đang được triển khai.  
Con đường phía trước:  
Báo cáo này khuyến nghị mạnh mẽ nên theo đuổi Hướng Nghiên cứu 1: Đánh giá Chuẩn các Mô hình XAI Bậc cao trong các Môi trường Giàu Tương tác.  
**Lý do:**

1. **Tác động Cao:** Hướng đi này là con đường trực tiếp và mạnh mẽ nhất để có được một công bố khoa học có tác động lớn. Nó có một giả thuyết rõ ràng và có thể kiểm chứng.  
2. **Tính Khả thi:** Nó tận dụng các công cụ hiện có (thư viện RL, các triển khai SHAP tiên tiến) và không yêu cầu phát minh ra các phương pháp hoàn toàn mới từ đầu.  
3. **Đóng góp Độc đáo:** Nó giải quyết một nhu cầu cấp thiết trong cộng đồng về việc đánh giá chặt chẽ các phương pháp XAI trong các miền phức tạp. Việc phát triển các thước đo độ trung thực dành riêng cho RL, tự nó, đã là một đóng góp đáng kể.

Công việc Tương lai:  
Các hướng đi 2 (Phản thực tế) và 3 (Nhân quả) nên được định hình là những bước tiếp theo thú vị và logic, có thể được xây dựng dựa trên các kết quả thực nghiệm nền tảng được thiết lập trong bài báo đầu tiên. Điều này định vị công việc ban đầu như là viên đá tảng của một chương trình nghiên cứu lớn hơn, có tầm ảnh hưởng sâu rộng trong lĩnh vực Học tăng cường Giải thích được.

#### **Nguồn trích dẫn**

1. A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME \- arXiv, truy cập vào tháng 9 23, 2025, [https://arxiv.org/html/2305.02012v3](https://arxiv.org/html/2305.02012v3)  
2. (PDF) A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME, truy cập vào tháng 9 23, 2025, [https://www.researchgate.net/publication/381774679\_A\_Perspective\_on\_Explainable\_Artificial\_Intelligence\_Methods\_SHAP\_and\_LIME](https://www.researchgate.net/publication/381774679_A_Perspective_on_Explainable_Artificial_Intelligence_Methods_SHAP_and_LIME)  
3. 18 SHAP – Interpretable Machine Learning, truy cập vào tháng 9 23, 2025, [https://christophm.github.io/interpretable-ml-book/shap.html](https://christophm.github.io/interpretable-ml-book/shap.html)  
4. SHAP and LIME Python Libraries: Part 1 \- Great Explainers, with Pros and Cons to Both, truy cập vào tháng 9 23, 2025, [https://domino.ai/blog/shap-lime-python-libraries-part-1-great-explainers-pros-cons](https://domino.ai/blog/shap-lime-python-libraries-part-1-great-explainers-pros-cons)  
5. LIME vs SHAP: A Comparative Analysis of Interpretability Tools \- MarkovML, truy cập vào tháng 9 23, 2025, [https://www.markovml.com/blog/lime-vs-shap](https://www.markovml.com/blog/lime-vs-shap)  
6. arxiv.org, truy cập vào tháng 9 23, 2025, [https://arxiv.org/html/2211.06665v5](https://arxiv.org/html/2211.06665v5)  
7. 28 Saliency Maps – Interpretable Machine Learning, truy cập vào tháng 9 23, 2025, [https://christophm.github.io/interpretable-ml-book/pixel-attribution.html](https://christophm.github.io/interpretable-ml-book/pixel-attribution.html)  
8. What is Explainable AI (XAI)? \- IBM, truy cập vào tháng 9 23, 2025, [https://www.ibm.com/think/topics/explainable-ai](https://www.ibm.com/think/topics/explainable-ai)  
9. XAI: Explainable Artificial Intelligence \- DARPA, truy cập vào tháng 9 23, 2025, [https://www.darpa.mil/research/programs/explainable-artificial-intelligence](https://www.darpa.mil/research/programs/explainable-artificial-intelligence)  
10. LIME vs SHAP: What's the Difference for Model Interpretability? \- ApX Machine Learning, truy cập vào tháng 9 23, 2025, [https://apxml.com/posts/lime-vs-shap-difference-interpretability](https://apxml.com/posts/lime-vs-shap-difference-interpretability)  
11. From Abstract to Actionable: Pairwise Shapley Values for Explainable AI \- arXiv, truy cập vào tháng 9 23, 2025, [https://arxiv.org/html/2502.12525v1](https://arxiv.org/html/2502.12525v1)  
12. Polynomial Features: A Comprehensive Guide (From Basics to Advanced) | by Adnan Mazraeh | Medium, truy cập vào tháng 9 23, 2025, [https://medium.com/@adnan.mazraeh1993/polynomial-features-a-comprehensive-guide-from-basics-to-advanced-5f18c430a137](https://medium.com/@adnan.mazraeh1993/polynomial-features-a-comprehensive-guide-from-basics-to-advanced-5f18c430a137)  
13. How to Use Polynomial Feature Transforms for Machine Learning \- MachineLearningMastery.com, truy cập vào tháng 9 23, 2025, [https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/](https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/)  
14. PolynomialFeatures — scikit-learn 1.7.2 documentation, truy cập vào tháng 9 23, 2025, [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)  
15. Identifiability of Deep Polynomial Neural Networks \- arXiv, truy cập vào tháng 9 23, 2025, [https://arxiv.org/html/2506.17093v1](https://arxiv.org/html/2506.17093v1)  
16. Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles, truy cập vào tháng 9 23, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/29352/30552](https://ojs.aaai.org/index.php/AAAI/article/view/29352/30552)  
17. shapiq: Shapley Interactions for Machine Learning, truy cập vào tháng 9 23, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/eb3a9313405e2d4175a5a3cfcd49999b-Paper-Datasets\_and\_Benchmarks\_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/eb3a9313405e2d4175a5a3cfcd49999b-Paper-Datasets_and_Benchmarks_Track.pdf)  
18. NeurIPS Poster shapiq: Shapley Interactions for Machine Learning, truy cập vào tháng 9 23, 2025, [https://neurips.cc/virtual/2024/poster/97533](https://neurips.cc/virtual/2024/poster/97533)  
19. \< Mul\&plica\&ve SHAP Values: Advancing Interpretable Machine Learning in General Insurance Pricing \>, truy cập vào tháng 9 23, 2025, [https://content.actuaries.asn.au/resources/resource-ce6yyqn64sx3-786882053-16123](https://content.actuaries.asn.au/resources/resource-ce6yyqn64sx3-786882053-16123)  
20. X-SHAP: towards multiplicative explainability of Machine Learning \- ResearchGate, truy cập vào tháng 9 23, 2025, [https://www.researchgate.net/publication/342027134\_X-SHAP\_towards\_multiplicative\_explainability\_of\_Machine\_Learning](https://www.researchgate.net/publication/342027134_X-SHAP_towards_multiplicative_explainability_of_Machine_Learning)  
21. mSHAP: SHAP Values for Two-Part Models \- MDPI, truy cập vào tháng 9 23, 2025, [https://www.mdpi.com/2227-9091/10/1/3](https://www.mdpi.com/2227-9091/10/1/3)  
22. mshap: Multiplicative SHAP Values for Two-Part Models \- CRAN, truy cập vào tháng 9 23, 2025, [https://cran.r-project.org/web/packages/mshap/mshap.pdf](https://cran.r-project.org/web/packages/mshap/mshap.pdf)  
23. Reinforcement learning \- Wikipedia, truy cập vào tháng 9 23, 2025, [https://en.wikipedia.org/wiki/Reinforcement\_learning](https://en.wikipedia.org/wiki/Reinforcement_learning)  
24. Model Learning to Solve Minecraft Tasks \- PRL Workshop Series, truy cập vào tháng 9 23, 2025, [https://prl-theworkshop.github.io/prl2023-icaps/papers/model-learning-minecraft-tasks.pdf](https://prl-theworkshop.github.io/prl2023-icaps/papers/model-learning-minecraft-tasks.pdf)  
25. A Deep Hierarchical Approach to Lifelong Learning in Minecraft \- Wix.com, truy cập vào tháng 9 23, 2025, [https://chentessler.wixsite.com/hdrlnminecraft](https://chentessler.wixsite.com/hdrlnminecraft)  
26. Learning to play Minecraft with Video PreTraining \- OpenAI, truy cập vào tháng 9 23, 2025, [https://openai.com/index/vpt/](https://openai.com/index/vpt/)  
27. Project Malmo \- Microsoft Research, truy cập vào tháng 9 23, 2025, [https://www.microsoft.com/en-us/research/project/project-malmo/](https://www.microsoft.com/en-us/research/project/project-malmo/)  
28. tsmatz/minecraft-rl-example: Applying Reinforcement Learning in Minecraft \- Project Malmo Tutorial (Mar 2021\) \- GitHub, truy cập vào tháng 9 23, 2025, [https://github.com/tsmatz/minecraft-rl-example](https://github.com/tsmatz/minecraft-rl-example)  
29. Reinforcement learning in Minecraft \- IDA.LiU.SE, truy cập vào tháng 9 23, 2025, [https://www.ida.liu.se/\~TDDE19/info/reports/2020/5.pdf](https://www.ida.liu.se/~TDDE19/info/reports/2020/5.pdf)  
30. MineDojo/Voyager: An Open-Ended Embodied Agent with Large Language Models \- GitHub, truy cập vào tháng 9 23, 2025, [https://github.com/MineDojo/Voyager](https://github.com/MineDojo/Voyager)  
31. Factorio Learning Env. | Epoch AI, truy cập vào tháng 9 23, 2025, [https://epoch.ai/benchmarks/factorio-learning-environment](https://epoch.ai/benchmarks/factorio-learning-environment)  
32. Factorio Learning Environment \- arXiv, truy cập vào tháng 9 23, 2025, [https://arxiv.org/html/2503.09617v1](https://arxiv.org/html/2503.09617v1)  
33. \[Literature Review\] Factorio Learning Environment \- Moonlight, truy cập vào tháng 9 23, 2025, [https://www.themoonlight.io/en/review/factorio-learning-environment](https://www.themoonlight.io/en/review/factorio-learning-environment)  
34. An autonomous agent framework for the game "Don't Starve" | Request PDF \- ResearchGate, truy cập vào tháng 9 23, 2025, [https://www.researchgate.net/publication/387481473\_An\_autonomous\_agent\_framework\_for\_the\_game\_Don't\_Starve](https://www.researchgate.net/publication/387481473_An_autonomous_agent_framework_for_the_game_Don't_Starve)  
35. Creating an Agent-Based Framework for Don't Starve Together \- GitHub, truy cập vào tháng 9 23, 2025, [https://raw.githubusercontent.com/hineios/dissertation/master/dissertation-paper/dissertation-paper.pdf](https://raw.githubusercontent.com/hineios/dissertation/master/dissertation-paper/dissertation-paper.pdf)  
36. \[Showcase\] Prototype Artificial Intelligence inside Terraria (pre-1.3.1), truy cập vào tháng 9 23, 2025, [https://forums.terraria.org/index.php?threads/showcase-prototype-artificial-intelligence-inside-terraria-pre-1-3-1.75890/](https://forums.terraria.org/index.php?threads/showcase-prototype-artificial-intelligence-inside-terraria-pre-1-3-1.75890/)  
37. dkoleber/TerrarAI: A platform for reinforcement learning in Terraria \- GitHub, truy cập vào tháng 9 23, 2025, [https://github.com/dkoleber/TerrarAI](https://github.com/dkoleber/TerrarAI)  
38. Starcraft II: A new challenge for reinforcement learning \- DocDrop, truy cập vào tháng 9 23, 2025, [https://arxiv.org/abs/1708.04782](https://arxiv.org/abs/1708.04782)  
39. Deep Reinforcement Learning in Starcraft II, truy cập vào tháng 9 23, 2025, [https://www.micsymposium.org/mics2019/wp-content/uploads/2019/05/SC2.pdf](https://www.micsymposium.org/mics2019/wp-content/uploads/2019/05/SC2.pdf)  
40. Brick-by-Brick: Combinatorial Construction with Deep Reinforcement ..., truy cập vào tháng 9 23, 2025, [https://proceedings.neurips.cc/paper/2021/file/2d4027d6df9c0256b8d4474ce88f8c88-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/2d4027d6df9c0256b8d4474ce88f8c88-Paper.pdf)  
41. We love 3D puzzle games like Portal and Talos Principle and fused it with creature-collecting in our game, Ryder. Our demo just went live\! : r/puzzlevideogames \- Reddit, truy cập vào tháng 9 23, 2025, [https://www.reddit.com/r/puzzlevideogames/comments/1icwqnd/we\_love\_3d\_puzzle\_games\_like\_portal\_and\_talos/](https://www.reddit.com/r/puzzlevideogames/comments/1icwqnd/we_love_3d_puzzle_games_like_portal_and_talos/)  
42. Just released The Trials 2 — a puzzle game inspired by Portal, The Witness, and The Talos Principle\! : r/puzzlevideogames \- Reddit, truy cập vào tháng 9 23, 2025, [https://www.reddit.com/r/puzzlevideogames/comments/1lj3lox/just\_released\_the\_trials\_2\_a\_puzzle\_game\_inspired/](https://www.reddit.com/r/puzzlevideogames/comments/1lj3lox/just_released_the_trials_2_a_puzzle_game_inspired/)  
43. Create a Custom Environment \- Gymnasium Documentation, truy cập vào tháng 9 23, 2025, [https://gymnasium.farama.org/introduction/create\_custom\_env/](https://gymnasium.farama.org/introduction/create_custom_env/)  
44. PettingZoo : Multi-Agent Reinforcement Learning \- GeeksforGeeks, truy cập vào tháng 9 23, 2025, [https://www.geeksforgeeks.org/deep-learning/pettingzoo-multi-agent-reinforcement-learning/](https://www.geeksforgeeks.org/deep-learning/pettingzoo-multi-agent-reinforcement-learning/)  
45. MPE \- PettingZoo Documentation, truy cập vào tháng 9 23, 2025, [https://pettingzoo.farama.org/environments/mpe/](https://pettingzoo.farama.org/environments/mpe/)  
46. PettingZoo: Gym for Multi-Agent Reinforcement Learning, truy cập vào tháng 9 23, 2025, [https://proceedings.neurips.cc/paper/2021/hash/7ed2d3454c5eea71148b11d0c25104ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7ed2d3454c5eea71148b11d0c25104ff-Abstract.html)  
47. DLR-RM/stable-baselines3: PyTorch version of Stable Baselines, reliable implementations of reinforcement learning algorithms. \- GitHub, truy cập vào tháng 9 23, 2025, [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)  
48. Stable-Baselines3 Docs \- Reliable Reinforcement Learning Implementations — Stable Baselines3 2.7.1a3 documentation, truy cập vào tháng 9 23, 2025, [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)  
49. shap/shap: A game theoretic approach to explain the output of any machine learning model. \- GitHub, truy cập vào tháng 9 23, 2025, [https://github.com/shap/shap](https://github.com/shap/shap)  
50. Explainable Reinforcement Learning \- Shapley \- Kaggle, truy cập vào tháng 9 23, 2025, [https://www.kaggle.com/code/kooaslansefat/explainable-reinforcement-learning-shapley](https://www.kaggle.com/code/kooaslansefat/explainable-reinforcement-learning-shapley)  
51. SHAP with DQN · Issue \#2384 \- GitHub, truy cập vào tháng 9 23, 2025, [https://github.com/slundberg/shap/issues/2384](https://github.com/slundberg/shap/issues/2384)  
52. Towards Faithful Model Explanation in NLP: A ... \- Veronica Qing Lyu, truy cập vào tháng 9 23, 2025, [https://veronica320.github.io/WPE2/survey.pdf](https://veronica320.github.io/WPE2/survey.pdf)  
53. A comprehensive analysis of perturbation methods in explainable AI feature attribution validation for neural time series classifiers \- PubMed Central, truy cập vào tháng 9 23, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12284047/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12284047/)  
54. XAI-TRIS: non-linear image benchmarks to quantify false positive ..., truy cập vào tháng 9 23, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11306297/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11306297/)  
55. $\\mathcal{M}^4$: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models | OpenReview, truy cập vào tháng 9 23, 2025, [https://openreview.net/forum?id=6zcfrSz98y¬eId=dhrMalPaBR](https://openreview.net/forum?id=6zcfrSz98y&noteId=dhrMalPaBR)  
56. GANterfactual-RL: Understanding Reinforcement Learning Agents ..., truy cập vào tháng 9 23, 2025, [https://www.southampton.ac.uk/\~eg/AAMAS2023/pdfs/p1097.pdf](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p1097.pdf)  
57. Counterfactual Explanations for Continuous Action Reinforcement Learning \- arXiv, truy cập vào tháng 9 23, 2025, [https://arxiv.org/html/2505.12701v1](https://arxiv.org/html/2505.12701v1)  
58. Robust Counterfactual Explanations in Machine Learning: A Survey \- IJCAI, truy cập vào tháng 9 23, 2025, [https://www.ijcai.org/proceedings/2024/0894.pdf](https://www.ijcai.org/proceedings/2024/0894.pdf)  
59. Explainable Reinforcement Learning through a Causal Lens \- AAAI, truy cập vào tháng 9 23, 2025, [https://cdn.aaai.org/ojs/5631/5631-13-8856-1-10-20200512.pdf](https://cdn.aaai.org/ojs/5631/5631-13-8856-1-10-20200512.pdf)  
60. Causal Reinforcement Learning, truy cập vào tháng 9 23, 2025, [https://crl.causalai.net/](https://crl.causalai.net/)  
61. Causal Influence Detection for Improving Efficiency in Reinforcement Learning \- NIPS, truy cập vào tháng 9 23, 2025, [https://proceedings.neurips.cc/paper/2021/file/c1722a7941d61aad6e651a35b65a9c3e-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/c1722a7941d61aad6e651a35b65a9c3e-Paper.pdf)