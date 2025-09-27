

# **Lộ trình Toàn diện trở thành Lập trình viên Next.js Chuyên nghiệp**

Chào mừng bạn đến với lộ trình chi tiết để trở thành một Lập trình viên Next.js chuyên nghiệp, được thiết kế đặc biệt cho những người bắt đầu từ con số không. Đây không chỉ là một danh sách các công nghệ cần học; đây là một con đường có cấu trúc, tập trung vào việc xây dựng một tư duy phát triển hiện đại, giúp bạn không chỉ viết code mà còn hiểu sâu sắc "tại sao" đằng sau mỗi dòng lệnh. Trong bối cảnh công nghệ năm 2025, Next.js đã khẳng định vị thế là một framework full-stack hàng đầu, xóa nhòa ranh giới truyền thống giữa frontend và backend. Lộ trình này sẽ trang bị cho bạn đầy đủ kỹ năng để làm chủ hệ sinh thái mạnh mẽ này, từ việc xây dựng giao diện tĩnh cho đến triển khai các ứng dụng web phức tạp, hiệu năng cao và sẵn sàng cho môi trường sản xuất. Hãy bắt đầu hành trình chinh phục thế giới phát triển web hiện đại.

## **Giai đoạn 0: Xây dựng Nền tảng Bất biến**

Giai đoạn đầu tiên này là quan trọng nhất, giống như việc xây dựng móng cho một tòa nhà chọc trời. Mục tiêu ở đây không phải là sự phức tạp, mà là sự vững chắc. Trước khi viết bất kỳ logic lập trình nào, một lập trình viên chuyên nghiệp phải thành thạo nghệ thuật cấu trúc và trình bày nội dung trên web. Giai đoạn này tập trung hoàn toàn vào việc tạo ra các trang web tĩnh, đẹp mắt, có cấu trúc tốt và có thể truy cập trên mọi thiết bị. Đây là nền tảng mà mọi kỹ năng sau này sẽ được xây dựng dựa trên.

### **Chủ đề chính: HTML5, CSS3, và Git**

#### **Các khái niệm cốt lõi cần nghiên cứu:**

* **HTML5:** Đây là bộ xương của mọi trang web. Việc học HTML không chỉ dừng lại ở việc biết các thẻ.  
  * **Cấu trúc tài liệu và các thẻ ngữ nghĩa (Semantic Tags):** Một trong những sai lầm phổ biến của người mới bắt đầu là lạm dụng thẻ \<div\> cho mọi thứ. Cách tiếp cận chuyên nghiệp đòi hỏi việc sử dụng các thẻ ngữ nghĩa. Các thẻ như \<header\>, \<nav\>, \<main\>, \<section\>, \<article\>, và \<footer\> không chỉ giúp nhóm các phần tử lại với nhau mà còn cung cấp ý nghĩa về cấu trúc của trang web cho cả trình duyệt, công cụ tìm kiếm và các công nghệ hỗ trợ như trình đọc màn hình.1 Việc hiểu rõ sự khác biệt giữa  
    \<section\> (dùng để nhóm các nội dung có cùng chủ đề) và \<article\> (dùng cho nội dung độc lập, có thể tự phân phối như một bài blog hoặc tin tức) là cực kỳ quan trọng để tránh nhầm lẫn và xây dựng cấu trúc trang web một cách chính xác.1 Việc sử dụng đúng thẻ ngữ nghĩa là nền tảng cơ bản cho Tối ưu hóa Công cụ tìm kiếm (SEO) và Khả năng truy cập (Accessibility), hai yếu tố không thể thiếu trong các dự án thực tế.  
  * **Forms và input:** Form là phương tiện chính để người dùng tương tác và gửi dữ liệu đến ứng dụng. Cần nắm vững cách sử dụng các thẻ \<form\>, \<input\> với các loại khác nhau (text, password, email, checkbox, radio), \<label\> để cải thiện khả năng truy cập, \<textarea\> cho nội dung dài và \<button\>.2  
  * **Các thẻ media:** Web hiện đại không thể thiếu hình ảnh, âm thanh và video. Cần học cách nhúng các nội dung đa phương tiện này một cách hiệu quả bằng các thẻ \<img\>, \<audio\>, và \<video\>.2  
* **CSS3:** Nếu HTML là bộ xương, thì CSS là làn da và quần áo, quyết định vẻ ngoài của trang web.  
  * **Box Model:** Đây là khái niệm nền tảng tuyệt đối của layout trong CSS. Mọi phần tử trên trang web có thể được coi là một chiếc hộp hình chữ nhật. Mô hình hộp này bao gồm bốn phần: content (nội dung), padding (vùng đệm bên trong), border (đường viền), và margin (lề bên ngoài).3 Một thiết lập quan trọng mà các lập trình viên hiện đại luôn sử dụng là  
    box-sizing: border-box;. Thiết lập này làm cho việc tính toán kích thước trở nên trực quan hơn, vì width và height sẽ bao gồm cả padding và border, thay vì chỉ content.3  
  * **Selectors:** Học cách nhắm mục tiêu chính xác các phần tử HTML để áp dụng style. Bắt đầu với các bộ chọn cơ bản (theo thẻ, class, ID) và tiến tới các pseudo-class quan trọng như :hover, :focus để tạo ra các hiệu ứng tương tác.  
  * **Flexbox và CSS Grid:** Đây là hai công cụ layout mạnh mẽ nhất trong CSS hiện đại. Việc hiểu khi nào nên dùng công cụ nào là dấu hiệu của sự chuyên nghiệp. Flexbox được thiết kế cho layout một chiều (một hàng *hoặc* một cột), lý tưởng cho việc căn chỉnh các mục trong một container như thanh điều hướng.4 Ngược lại, CSS Grid được thiết kế cho layout hai chiều (cả hàng  
    *và* cột cùng một lúc), hoàn hảo cho việc xây dựng các bố cục phức tạp như toàn bộ trang web.4  
  * **Responsive Design (Mobile-First):** Đây không chỉ là một kỹ thuật, mà là một triết lý thiết kế. Thay vì thiết kế cho màn hình lớn rồi thu nhỏ lại, phương pháp Mobile-First yêu cầu thiết kế cho màn hình nhỏ nhất (điện thoại di động) trước tiên, sau đó sử dụng media queries để "tăng cường dần" (progressive enhancement) cho các màn hình lớn hơn như máy tính bảng và máy tính để bàn.6 Cách tiếp cận này buộc người phát triển phải tập trung vào những nội dung và chức năng cốt lõi nhất, dẫn đến trải nghiệm người dùng tốt hơn trên di động và cải thiện hiệu suất tải trang.6  
  * **Các đơn vị:** Hiểu sự khác biệt giữa các đơn vị tuyệt đối như px và các đơn vị tương đối như rem, em, %, vw, vh. Trong phát triển hiện đại, rem (root em) thường được ưa chuộng cho kích thước phông chữ và khoảng cách để tạo ra các giao diện có thể co giãn và dễ tiếp cận hơn.  
* **Git & GitHub:** Đây là công cụ và nền tảng không thể thiếu cho bất kỳ lập trình viên nào.  
  * **Workflow cơ bản:** Git không phải là một công việc vặt, nó là mạng lưới an toàn quan trọng nhất của bạn. Cần thành thạo chu trình làm việc cốt lõi: git clone để lấy mã nguồn về máy, git add để đưa các thay đổi vào khu vực chờ, git commit \-m "thông điệp có ý nghĩa" để lưu lại một "ảnh chụp" của các thay đổi, và git push để đẩy các thay đổi đó lên kho lưu trữ từ xa trên GitHub.7 Một lịch sử commit rõ ràng giống như một cỗ máy thời gian, cho phép bạn hoàn tác các lỗi một cách không sợ hãi.  
  * **Khái niệm về branch và pull request:** Branching (phân nhánh) cho phép bạn làm việc trên các tính năng mới một cách độc lập mà không ảnh hưởng đến mã nguồn chính (thường là nhánh main hoặc master). Pull Request (PR) là một yêu cầu chính thức để hợp nhất nhánh tính năng của bạn vào nhánh chính. Đây là cơ chế trung tâm cho việc hợp tác và review code trong các đội ngũ phát triển chuyên nghiệp.

#### **Mục tiêu cần đạt được:**

Có khả năng tự xây dựng một trang web tĩnh, responsive hoàn chỉnh, có cấu trúc ngữ nghĩa tốt và quản lý mã nguồn một cách chuyên nghiệp trên GitHub.

#### **Dự án thực hành gợi ý:**

Xây dựng một trang portfolio cá nhân gồm 3 trang (Giới thiệu, Dự án, Liên hệ). Trang web phải responsive hoàn toàn trên mobile, tablet, và desktop, sử dụng Flexbox và/hoặc Grid cho layout, và được đăng tải lên GitHub Pages để chia sẻ với thế giới.

#### **Tài liệu chi tiết:**

[[01-nen-tang-bat-bien.md]]

---

## **Giai đoạn 1: Nắm vững JavaScript Hiện đại**

Sau khi đã làm chủ được việc trình bày nội dung tĩnh, Giai đoạn 1 sẽ đưa bạn vào thế giới của sự tương tác và logic. JavaScript là ngôn ngữ lập trình làm cho web trở nên sống động. Giai đoạn này tập trung vào việc nắm vững các tính năng của JavaScript hiện đại (ES6+), những tính năng này là điều kiện tiên quyết tuyệt đối để học React và Next.js. Đây là lúc bạn chuyển từ một "nhà thiết kế web" thành một "lập trình viên web".

### **Chủ đề chính: JavaScript (ES6+)**

#### **Các khái niệm cốt lõi cần nghiên cứu:**

* **Nền tảng:**  
  * **Biến (let, const), kiểu dữ liệu, hàm:** Cần hiểu sâu sắc sự khác biệt giữa var (có cơ chế hoisting và phạm vi hàm gây khó hiểu), let (có phạm vi khối và có thể gán lại giá trị), và const (có phạm vi khối và không thể gán lại).8 Quy tắc vàng của lập trình hiện đại là: luôn dùng  
    const theo mặc định, chỉ dùng let khi biến đó thực sự cần được gán lại giá trị. Nắm vững các kiểu dữ liệu cơ bản (primitive types) và đối tượng (objects). Đặc biệt, cần thành thạo cú pháp **arrow functions (=\>)**, vì nó được sử dụng ở khắp mọi nơi trong code React hiện đại do sự ngắn gọn và cách xử lý từ khóa this nhất quán.8  
  * **Scope:** Hiểu rõ về Global Scope, Function Scope, và Block Scope để tránh các lỗi phổ biến liên quan đến việc truy cập biến.  
* **Cấu trúc dữ liệu:**  
  * **Thao tác chuyên sâu với Objects và Arrays:** Vượt ra ngoài các thao tác cơ bản. Cần tập trung vào các phương thức bậc cao của mảng, vì chúng là nền tảng của phong cách lập trình khai báo (declarative programming) trong React. Cụ thể:  
    * .map(): Dùng để biến đổi một mảng dữ liệu thành một mảng các phần tử giao diện (JSX). Đây là phương thức được sử dụng nhiều nhất để hiển thị danh sách trong React.9  
    * .filter(): Dùng để tạo ra một mảng mới chứa các phần tử thỏa mãn một điều kiện nhất định, hữu ích cho việc tìm kiếm hoặc lọc dữ liệu.9  
    * .reduce(): Dùng cho các tác vụ tổng hợp dữ liệu phức tạp hơn, biến một mảng thành một giá trị duy nhất (ví dụ: tính tổng giá trị giỏ hàng).9

      Việc thành thạo các phương thức này chính là học "ngôn ngữ" của React.  
* **Bất đồng bộ:**  
  * **Promises và cú pháp async/await:** Đây là một trong những khái niệm khó nhất đối với người mới bắt đầu nhưng lại cực kỳ quan trọng. Hầu hết các ứng dụng web đều cần tương tác với máy chủ để lấy dữ liệu, và các hoạt động này là bất đồng bộ. Cần hiểu Promise là một đối tượng đại diện cho sự thành công hay thất bại của một hoạt động bất đồng bộ trong tương lai. Sau đó, học async/await như là một "lớp vỏ cú pháp" hiện đại giúp viết code bất đồng bộ trông giống như code đồng bộ, tuần tự, làm cho nó dễ đọc và dễ quản lý hơn rất nhiều.10 Việc sử dụng khối  
    try...catch để xử lý lỗi trong các hàm async là một kỹ năng bắt buộc.10  
* **ES6+:**  
  * **Destructuring:** Học cách "giải nén" các giá trị từ mảng hoặc thuộc tính từ đối tượng vào các biến riêng biệt. Kỹ thuật này được sử dụng liên tục trong React để nhận props và quản lý state.8  
  * **Spread/Rest operators (...):** Toán tử Spread (...) được dùng để tạo ra các bản sao nông (shallow copies) của mảng và đối tượng. Đây là nền tảng của nguyên tắc **bất biến (immutability)** \- một khái niệm cốt lõi trong React. Thay vì thay đổi trực tiếp dữ liệu gốc, chúng ta tạo ra một bản sao mới với các thay đổi. Điều này giúp React phát hiện sự thay đổi trạng thái một cách đáng tin cậy và kích hoạt re-render. Toán tử Rest được dùng để gom nhiều phần tử lại thành một mảng, thường dùng trong tham số hàm.8  
  * **Modules (import/export):** Hiểu cách chia nhỏ code thành các tệp (modules) có thể tái sử dụng và cách nhập (import) và xuất (export) chúng. Đây là nền tảng của kiến trúc dựa trên component.8  
* **Tương tác DOM:**  
  * Trước khi nhảy vào React, cần có một hiểu biết cơ bản về cách JavaScript tương tác trực tiếp với DOM (Document Object Model). Học các phương thức cơ bản như document.querySelector(), element.addEventListener(), và element.textContent. Mục tiêu không phải là để thành thạo thao tác DOM thủ công, mà là để hiểu rõ React đang tự động hóa và trừu tượng hóa công việc gì cho chúng ta.11 Bối cảnh này sẽ làm cho giá trị của React trở nên rõ ràng hơn rất nhiều.

#### **Mục tiêu cần đạt được:**

Có khả năng xây dựng các ứng dụng web tương tác phía client, xử lý logic, thao tác với cấu trúc dữ liệu phức tạp, và gọi API để lấy và hiển thị dữ liệu động.

#### **Dự án thực hành gợi ý:**

Xây dựng một ứng dụng "Pokedex" (danh bạ Pokémon) đơn giản. Ứng dụng này sẽ gọi đến một API công khai (ví dụ: PokeAPI), cho phép người dùng tìm kiếm Pokémon theo tên, và hiển thị hình ảnh cùng thông tin chi tiết. Dự án này sẽ củng cố kỹ năng về async/await để gọi API, thao tác với mảng/đối tượng (dùng .map để hiển thị danh sách kết quả), và xử lý sự kiện của người dùng.

#### **Tài liệu chi tiết:**

[[02-javascript-hien-dai.md]]

---

## **Giai đoạn 2: Tư duy với React**

Giai đoạn này là một bước nhảy vọt về tư duy. Thay vì viết các kịch bản tuần tự để thao tác trang web, bạn sẽ học cách xây dựng giao diện người dùng (UI) như một cây gồm các khối độc lập, có thể tái sử dụng và có trạng thái riêng, được gọi là **components**. "Tư duy với React" là nghệ thuật phân rã một giao diện phức tạp thành những mảnh ghép nhỏ, dễ quản lý. Đây là lúc bạn thực sự bắt đầu xây dựng các ứng dụng web hiện đại.

### **Chủ đề chính: React.js**

#### **Các khái niệm cốt lõi cần nghiên cứu:**

* **JSX, Components, Props:**  
  * **JSX:** Là một phần mở rộng cú pháp cho JavaScript, cho phép viết mã trông giống HTML ngay trong các tệp JavaScript. Nó giúp việc mô tả UI trở nên trực quan và quen thuộc.  
  * **Components:** Là trái tim của React. Một component là một hàm JavaScript trả về JSX, đóng gói một phần của UI và logic liên quan. Ví dụ, một trang web có thể được chia thành các component như Header, Sidebar, ArticleList, Footer.  
  * **Props (Properties):** Là cách để truyền dữ liệu từ component cha xuống component con. Props là "chỉ đọc" (read-only), giúp các component trở nên linh hoạt và có thể tái sử dụng. Ví dụ, một component UserProfile có thể nhận name và avatarUrl làm props.  
* **State và React Hooks:**  
  * **State:** Là khái niệm quan trọng nhất trong React. State là dữ liệu mà một component "sở hữu" và có thể thay đổi theo thời gian dựa trên tương tác của người dùng. Khi state của một component thay đổi, React sẽ tự động **re-render** (vẽ lại) component đó và các component con của nó để phản ánh trạng thái mới. Toàn bộ công việc của React có thể được tóm gọn là một cỗ máy đồng bộ hóa UI với state.  
  * **React Hooks:** Là các hàm cho phép bạn "móc" vào các tính năng của React từ các functional components, như state và vòng đời, mà không cần viết class.12 Đây là cách viết React hiện đại.  
    * **useState:** Là Hook cơ bản nhất để thêm state vào một component. Nó trả về một mảng gồm hai phần tử: giá trị state hiện tại và một hàm để cập nhật giá trị đó.13 Khi gọi hàm cập nhật, bạn phải tuân thủ nguyên tắc bất biến đã học ở Giai đoạn 1\.  
    * **useEffect:** Là Hook để xử lý các "side effects" \- những hành động tương tác với thế giới bên ngoài component, như gọi API, đăng ký các sự kiện, hay thao tác DOM trực tiếp.12  
      useEffect nhận một hàm (effect) và một mảng phụ thuộc (dependency array). Mảng này quyết định khi nào effect sẽ chạy:  
      * Không có mảng phụ thuộc: Chạy sau mỗi lần render.  
      * Mảng rỗng \`\`: Chỉ chạy một lần sau lần render đầu tiên (tương đương componentDidMount trong class).15  
      * Mảng có giá trị \[value1, value2\]: Chạy sau lần render đầu tiên và mỗi khi một trong các giá trị trong mảng thay đổi (tương đương componentDidUpdate).15  
    * Một điểm mạnh của useEffect là nó cho phép nhóm các logic liên quan (ví dụ: logic đăng ký và hủy đăng ký một sự kiện) lại với nhau trong cùng một nơi, thay vì phải tách chúng ra các phương thức vòng đời khác nhau như trong class components, giúp code dễ đọc và bảo trì hơn.12  
* **Xử lý sự kiện (Event Handling) và Form:** Học cách xử lý các sự kiện của người dùng như onClick, onChange trong JSX. Một mẫu hình phổ biến và quan trọng là "controlled components", trong đó giá trị của các phần tử form (như \<input\>) được liên kết trực tiếp với một state của React. Mỗi khi người dùng gõ, sự kiện onChange được kích hoạt, cập nhật state, và component re-render với giá trị mới trong input.  
* **Vòng đời Component (Component Lifecycle):** Với Hooks, vòng đời của component được hiểu một cách đơn giản hơn:  
  * **Mounting (Gắn kết):** Khi component được tạo và chèn vào DOM lần đầu tiên.  
  * **Updating (Cập nhật):** Khi component re-render do props hoặc state thay đổi.  
  * Unmounting (Gỡ bỏ): Khi component bị xóa khỏi DOM.  
    useEffect là công cụ chính để thực hiện các hành động trong các giai đoạn này.14  
* **Context API:** Đây là giải pháp tích hợp sẵn của React để giải quyết vấn đề "prop drilling" \- việc phải truyền props qua nhiều cấp component trung gian không cần đến chúng. Context cho phép bạn tạo ra một "nguồn" dữ liệu toàn cục (Provider) và bất kỳ component nào trong cây con của nó đều có thể "tiêu thụ" (consume) dữ liệu đó. Nó rất phù hợp cho các state toàn cục ít thay đổi như thông tin người dùng đã đăng nhập, chủ đề (theme) sáng/tối.16 Tuy nhiên, nó không được tối ưu cho các cập nhật state tần suất cao, vì mọi component tiêu thụ context sẽ re-render khi giá trị context thay đổi, có thể gây ra vấn đề về hiệu năng.17

#### **Mục tiêu cần đạt được:**

Nắm vững tư duy component-based, có khả năng phân rã một UI phức tạp thành các component nhỏ, tái sử dụng được, quản lý state và side effects một cách hiệu quả, và xây dựng được một ứng dụng trang đơn (Single-Page Application \- SPA) hoàn chỉnh.

#### **Dự án thực hành gợi ý:**

Xây dựng một ứng dụng ghi chú (Note-taking app) hoặc To-do list. Ứng dụng phải cho phép người dùng thêm, sửa, xóa và đánh dấu hoàn thành các mục. Dự án này sẽ củng cố toàn bộ kiến thức về Components, Props, State (useState), Side Effects (useEffect để lưu dữ liệu vào localStorage của trình duyệt), và xử lý form.

#### **Tài liệu chi tiết:**

[[03-tu-duy-voi-react.md]]

---

## **Giai đoạn 3: Chinh phục Next.js (App Router)**

Đây là giai đoạn cốt lõi của lộ trình, nơi bạn chuyển từ việc xây dựng các ứng dụng chỉ chạy phía client (SPA) sang các ứng dụng full-stack, hiệu năng cao với Next.js. Giai đoạn này tập trung hoàn toàn vào **App Router**, mô hình mới và mạnh mẽ được giới thiệu từ Next.js 13, đại diện cho một sự thay đổi mô hình phát triển web. Thay vì mặc định mọi thứ diễn ra trên trình duyệt, App Router đưa logic trở lại máy chủ, kết hợp những ưu điểm tốt nhất của render phía máy chủ truyền thống và sự tương tác phong phú của các ứng dụng hiện đại.

### **Chủ đề chính: Next.js 14+ với App Router**

#### **Các khái niệm cốt lõi cần nghiên cứu:**

* **Routing:** Hệ thống định tuyến của App Router dựa trên hệ thống tệp.  
  * Một thư mục trong app tạo ra một phân đoạn URL. Ví dụ, app/dashboard/settings/page.tsx sẽ tương ứng với URL /dashboard/settings.  
  * page.tsx: Tệp này định nghĩa giao diện người dùng duy nhất cho một tuyến đường.18  
  * layout.tsx: Tệp này định nghĩa một giao diện chung được chia sẻ cho một phân đoạn và tất cả các tuyến đường con của nó. Rất hữu ích cho việc tạo header, footer, sidebar chung.19  
  * **Route động:** Bằng cách đặt tên thư mục trong dấu ngoặc vuông, ví dụ app/blog/\[slug\]/page.tsx, bạn có thể tạo ra các trang cho vô số bài blog khác nhau.  
  * **Route lồng nhau:** Cấu trúc thư mục lồng nhau tự động tạo ra các tuyến đường lồng nhau.  
* **Rendering: Server Components vs. Client Components:** Đây là khái niệm mang tính cách mạng và quan trọng nhất cần nắm vững.  
  * **Server Components (Mặc định):** Theo mặc định, tất cả các component trong App Router là Server Components.19 Chúng chỉ chạy trên máy chủ, không bao giờ chạy trên trình duyệt. Điều này mang lại những lợi ích to lớn:  
    * **Giảm kích thước JavaScript phía client:** Mã của Server Components không được gửi đến trình duyệt, giúp trang web tải nhanh hơn.19  
    * **Truy cập trực tiếp tài nguyên backend:** Chúng có thể async/await để lấy dữ liệu trực tiếp từ cơ sở dữ liệu, hệ thống tệp hoặc API nội bộ mà không cần tạo thêm một lớp API.20  
    * **Bảo mật:** Các thông tin nhạy cảm như API keys, token, hoặc logic truy cập cơ sở dữ liệu được giữ an toàn trên máy chủ.19  
  * **Client Components ('use client'):** Để thêm tính tương tác (sử dụng useState, useEffect, xử lý sự kiện onClick, truy cập API của trình duyệt như localStorage), bạn cần chỉ định một component là Client Component bằng cách đặt chuỗi 'use client' ở đầu tệp.21 Các component này ban đầu được render trên máy chủ để tạo HTML tĩnh (giúp tải trang ban đầu nhanh), sau đó được "hydrate" trên trình duyệt để trở nên tương tác.19  
  * **Mô hình làm việc:** Mô hình phổ biến và hiệu quả nhất là: sử dụng Server Components để fetch dữ liệu và xử lý logic nặng, sau đó truyền dữ liệu đó dưới dạng props cho các Client Components nhỏ hơn, chuyên trách xử lý tương tác người dùng.  
* **Data Fetching:** Trong App Router, việc lấy dữ liệu được đơn giản hóa tối đa. Bạn có thể biến một Server Component thành hàm async và dùng await để gọi fetch hoặc truy vấn cơ sở dữ liệu trực tiếp.20 Next.js mở rộng hàm  
  fetch gốc của web để cung cấp các chiến lược caching mạnh mẽ:  
  * **Mặc định (Static Rendering):** Next.js tự động cache kết quả của các yêu cầu fetch. Điều này tương tự như Static Site Generation (SSG), giúp các trang tải cực nhanh.20  
  * **Không cache (Server-side Rendering):** Bằng cách thêm tùy chọn { cache: 'no-store' } vào fetch, bạn yêu cầu Next.js lấy dữ liệu mới cho mỗi yêu cầu, tương tự như Server-Side Rendering (SSR).  
  * **Revalidating (Incremental Static Regeneration):** Tùy chọn { next: { revalidate: 60 } } cho phép bạn cache dữ liệu nhưng tự động làm mới nó sau một khoảng thời gian nhất định (ví dụ: 60 giây), kết hợp tốc độ của trang tĩnh và sự tươi mới của dữ liệu động.  
* **Server Actions:** Đây là cách tiếp cận hiện đại để xử lý các hành động thay đổi dữ liệu (mutations) như tạo, cập nhật, xóa. Server Actions là các hàm async được đánh dấu bằng 'use server' và có thể được gọi trực tiếp từ các component (thường là từ một \<form\>).22 Chúng chạy an toàn trên máy chủ, loại bỏ nhu cầu phải tự tạo các API endpoints riêng cho các tác vụ này. Chúng cũng tích hợp sẵn với cơ chế caching của Next.js, cho phép bạn dễ dàng làm mới dữ liệu trên trang sau khi một hành động hoàn tất (ví dụ: dùng  
  revalidatePath).22  
* **Tối ưu hóa:**  
  * **Component \<Image\>:** Component này là một công cụ tối ưu hóa hình ảnh tự động. Nó sẽ tự động thay đổi kích thước, nén, chuyển đổi sang các định dạng hiện đại như WebP hoặc AVIF, và lazy-load hình ảnh (chỉ tải khi chúng lọt vào khung nhìn). Điều này cải thiện đáng kể hiệu suất tải trang và ngăn chặn Cumulative Layout Shift (CLS).24  
  * **Component \<Link\>:** Dùng thay cho thẻ \<a\> để điều hướng giữa các trang trong ứng dụng. Nó cho phép điều hướng phía client, nghĩa là trang chuyển đổi gần như tức thì mà không cần tải lại toàn bộ trang. Nó cũng tự động tìm nạp trước (prefetch) mã của các trang liên kết trong nền, làm cho việc điều hướng trở nên nhanh hơn nữa.  
  * **Font Optimization:** next/font là công cụ giúp tối ưu hóa việc tải font, tránh các vấn đề về hiệu suất và layout shift liên quan đến font chữ.  
* **Route Handlers:** Mặc dù Server Actions là lựa chọn ưu tiên cho các mutation trong ứng dụng, đôi khi bạn vẫn cần tạo các API endpoints truyền thống (ví dụ: để cung cấp dữ liệu cho một ứng dụng di động hoặc một dịch vụ bên thứ ba). Route Handlers, được định nghĩa trong tệp route.ts (hoặc .js), cho phép bạn làm điều này bằng cách xuất các hàm tương ứng với các phương thức HTTP như GET, POST, PUT, DELETE.26

#### **Mục tiêu cần đạt được:**

Có khả năng xây dựng một ứng dụng web full-stack, hiệu năng cao, được tối ưu cho SEO bằng cách tận dụng sức mạnh của Server Components, Server Actions và các tính năng tối ưu hóa tích hợp của Next.js.

#### **Dự án thực hành gợi ý:**

Xây dựng một trang blog hoàn chỉnh. Trang chủ (Server Component) sẽ fetch và hiển thị danh sách các bài viết. Khi click vào một bài viết, người dùng sẽ được điều hướng đến một trang chi tiết (dynamic route) cũng được render trên server. Sẽ có một trang liên hệ với một form sử dụng Server Actions để gửi dữ liệu. Tất cả hình ảnh sẽ được tối ưu hóa bằng component \<Image\>.

#### **Tài liệu chi tiết:**

[[04-chinh-phuc-nextjs.md]]

---

## **Giai đoạn 4: Mở rộng Hệ sinh thái Chuyên nghiệp**

Sau khi đã làm chủ được Next.js, giai đoạn cuối cùng là trang bị cho mình bộ công cụ hoàn chỉnh để xây dựng các ứng dụng ở cấp độ sản xuất. Giai đoạn này tập trung vào việc tích hợp các thư viện và dịch vụ hàng đầu trong hệ sinh thái phát triển web hiện đại. Việc thành thạo những công cụ này không chỉ giúp tăng tốc độ phát triển mà còn đảm bảo ứng dụng của bạn có khả năng mở rộng, dễ bảo trì và mang lại trải nghiệm người dùng tuyệt vời. Đây là bước để chuyển từ một người biết Next.js thành một kỹ sư phần mềm full-stack thực thụ.

### **Chủ đề chính: Các công cụ và kỹ năng bổ trợ**

#### **Các khái niệm cốt lõi cần nghiên cứu:**

* **Styling: Tailwind CSS:**  
  * Tailwind CSS là một framework CSS theo triết lý **utility-first**. Thay vì cung cấp các component dựng sẵn (như Button, Card trong Bootstrap), Tailwind cung cấp một bộ lớn các class tiện ích cấp thấp, mỗi class thực hiện một chức năng duy nhất (ví dụ: pt-4 để thêm padding-top, flex để áp dụng display: flex, text-lg để tăng kích thước chữ).27 Bạn sẽ kết hợp các class này trực tiếp trong mã HTML/JSX của mình để xây dựng giao diện. Cách tiếp cận này giúp phát triển nhanh chóng, duy trì sự nhất quán trong thiết kế và loại bỏ gần như hoàn toàn việc phải viết CSS tùy chỉnh.29  
* **UI Libraries: Shadcn/ui:**  
  * Đây là một khái niệm đột phá và rất phổ biến trong hệ sinh thái Next.js. Shadcn/ui **không phải là một thư viện component** truyền thống mà bạn cài đặt qua npm.31 Thay vào đó, nó là một bộ sưu tập các component được thiết kế đẹp mắt, có khả năng truy cập cao (xây dựng trên Radix UI) và được style bằng Tailwind CSS. Bạn sử dụng công cụ dòng lệnh (CLI) của nó để  
    **sao chép mã nguồn** của từng component (ví dụ: Button, Dialog, Form) trực tiếp vào dự án của mình.33 Điều này mang lại cho bạn toàn quyền sở hữu và kiểm soát mã nguồn, cho phép bạn tùy chỉnh component theo bất kỳ cách nào bạn muốn mà không bị giới hạn bởi API của một thư viện bên ngoài.34 Đây là một kiến trúc component hiện đại, kết hợp sức mạnh của Radix (cho hành vi và khả năng truy cập) và Tailwind (cho giao diện).  
* **State Management: Zustand:**  
  * Khi ứng dụng phát triển, việc quản lý state phía client có thể trở nên phức tạp. Mặc dù Context API hữu ích cho các state toàn cục đơn giản, nó có thể gây ra các vấn đề về hiệu năng với các cập nhật thường xuyên. Zustand là một thư viện quản lý state nhỏ, nhanh và không có nhiều quy tắc ràng buộc.35 Nó cung cấp một giải pháp đơn giản dựa trên hooks để tạo ra các "store" chứa state và các hành động (actions) để thay đổi state đó. Bạn có thể sử dụng store này trong bất kỳ component nào mà không cần đến các  
    Provider bao bọc, giúp giảm đáng kể mã boilerplate.37 Zustand được tối ưu hóa để chỉ re-render các component khi phần state mà chúng thực sự sử dụng thay đổi, mang lại hiệu năng vượt trội.38

**Bảng 1: So sánh các giải pháp Quản lý State trong React/Next.js**

| Giải pháp (Solution) | Boilerplate (Độ rườm rà) | Độ dễ sử dụng (Ease of Use) | Hiệu năng (Performance) | Trường hợp sử dụng tốt nhất (Best Use Case) |
| :---- | :---- | :---- | :---- | :---- |
| **Context API** | Thấp | Dễ | Tốt cho state ít thay đổi | Global state đơn giản (theme, auth status) 16 |
| **Zustand** | Rất thấp | Rất dễ | Tối ưu cao (tránh re-render không cần thiết) | Client state từ đơn giản đến phức tạp, cần hiệu năng cao 36 |
| **Redux Toolkit** | Trung bình | Khó hơn | Tối ưu cao | Ứng dụng quy mô rất lớn, state phức tạp, cần DevTools mạnh mẽ 17 |

\<br\>

* **Database & ORM:**  
  * **Supabase (như một BaaS):** Supabase là một nền tảng Backend-as-a-Service (BaaS) mã nguồn mở, được xem là một sự thay thế mạnh mẽ cho Firebase. Nó cung cấp cho bạn một cơ sở dữ liệu PostgreSQL đầy đủ chức năng, hệ thống xác thực người dùng, lưu trữ tệp, và tự động tạo ra các API RESTful ngay từ schema cơ sở dữ liệu của bạn.40 Việc sử dụng Supabase giúp bạn có một backend hoàn chỉnh trong vài phút, cho phép bạn tập trung vào việc xây dựng ứng dụng thay vì quản lý cơ sở hạ tầng.42  
  * **Prisma (như một ORM):** Prisma là một Object-Relational Mapper (ORM) thế hệ mới dành cho Node.js và TypeScript. Nó cho phép bạn tương tác với cơ sở dữ liệu (như PostgreSQL của Supabase) bằng cách sử dụng các đối tượng và phương thức JavaScript/TypeScript một cách tự nhiên, thay vì phải viết các chuỗi truy vấn SQL thô.44 Điểm mạnh lớn nhất của Prisma là sự an toàn về kiểu (type-safety). Dựa trên schema cơ sở dữ liệu của bạn, Prisma Client được tạo ra sẽ cung cấp tính năng tự động hoàn thành (autocomplete) và kiểm tra kiểu tĩnh, giúp bắt lỗi ngay tại thời điểm phát triển, thay vì lúc chạy ứng dụng.46 Việc kết hợp Prisma với Server Components và Server Actions trong Next.js tạo ra một luồng làm việc full-stack liền mạch và hiệu quả.  
* **Authentication: Auth.js (trước đây là NextAuth.js):**  
  * Xác thực là một phần không thể thiếu của hầu hết các ứng dụng. Auth.js là giải pháp tiêu chuẩn và toàn diện nhất cho việc xác thực trong Next.js. Nó đơn giản hóa việc triển khai các luồng xác thực phức tạp, bao gồm đăng nhập bằng email/mật khẩu, cũng như đăng nhập qua các nhà cung cấp OAuth bên thứ ba như Google, GitHub, Facebook.48 Auth.js xử lý việc quản lý session, cookies, và các token một cách an toàn, tích hợp mượt mà với cả Server Components và Client Components.  
* **Deployment: Vercel:**  
  * Vercel là công ty đứng sau Next.js và là nền tảng được tối ưu hóa nhất để triển khai các ứng dụng Next.js. Nó cung cấp một trải nghiệm triển khai không cần cấu hình (zero-configuration). Chỉ cần kết nối kho lưu trữ GitHub của bạn, Vercel sẽ tự động build và triển khai ứng dụng của bạn mỗi khi bạn đẩy code lên. Nó cung cấp các tính năng mạnh mẽ như Mạng phân phối nội dung (CDN) toàn cầu, Serverless Functions, và các môi trường preview cho mỗi pull request, giúp quá trình phát triển và phát hành sản phẩm trở nên cực kỳ hiệu quả.

#### **Mục tiêu cần đạt được:**

Hoàn thiện bộ kỹ năng để xây dựng và triển khai một sản phẩm web full-stack hoàn chỉnh, an toàn, có khả năng mở rộng và sẵn sàng cho môi trường production.

#### **Dự án thực hành gợi ý:**

Phát triển một ứng dụng E-commerce nhỏ. Ứng dụng sẽ có các tính năng:

1. **Xác thực:** Người dùng có thể đăng ký và đăng nhập bằng email/mật khẩu hoặc tài khoản Google (sử dụng Auth.js).  
2. **Hiển thị sản phẩm:** Dữ liệu sản phẩm được lưu trữ trong cơ sở dữ liệu Supabase và được truy vấn bằng Prisma trong các Server Components.  
3. **Giao diện:** Giao diện người dùng được xây dựng bằng Next.js, Tailwind CSS, và các component có sẵn từ Shadcn/ui.  
4. **Quản lý giỏ hàng:** Trạng thái giỏ hàng phía client được quản lý bằng Zustand.  
5. **Triển khai:** Ứng dụng cuối cùng sẽ được triển khai lên Vercel để mọi người có thể truy cập.

#### **Tài liệu chi tiết:**

[[05-he-sinh-thai-chuyen-nghiep.md]]

#### **Nguồn trích dẫn**

1. Semantic HTML5 Elements Explained \- freeCodeCamp, truy cập vào tháng 8 23, 2025, [https://www.freecodecamp.org/news/semantic-html5-elements/](https://www.freecodecamp.org/news/semantic-html5-elements/)  
2. HTML5 Semantics \- GeeksforGeeks, truy cập vào tháng 8 23, 2025, [https://www.geeksforgeeks.org/html/html5-semantics/](https://www.geeksforgeeks.org/html/html5-semantics/)  
3. The CSS Box Model: Explained for Beginners \- Udacity, truy cập vào tháng 8 23, 2025, [https://www.udacity.com/blog/2021/04/the-css-box-model-explained-for-beginners.html](https://www.udacity.com/blog/2021/04/the-css-box-model-explained-for-beginners.html)  
4. Relationship of grid layout to other layout methods \- CSS \- MDN, truy cập vào tháng 8 23, 2025, [https://developer.mozilla.org/en-US/docs/Web/CSS/CSS\_grid\_layout/Relationship\_of\_grid\_layout\_with\_other\_layout\_methods](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout/Relationship_of_grid_layout_with_other_layout_methods)  
5. css \- What are the differences between flexbox and the grid systems? \- Stack Overflow, truy cập vào tháng 8 23, 2025, [https://stackoverflow.com/questions/50094544/what-are-the-differences-between-flexbox-and-the-grid-systems](https://stackoverflow.com/questions/50094544/what-are-the-differences-between-flexbox-and-the-grid-systems)  
6. Mobile First Design: What it is & How to implement it | BrowserStack, truy cập vào tháng 8 23, 2025, [https://www.browserstack.com/guide/how-to-implement-mobile-first-design](https://www.browserstack.com/guide/how-to-implement-mobile-first-design)  
7. Basic Git Workflow, truy cập vào tháng 8 23, 2025, [https://uidaholib.github.io/get-git/3workflow.html](https://uidaholib.github.io/get-git/3workflow.html)  
8. JavaScript ES6+ Features: A Beginner's Guide to Modern JavaScript | by Rashmi Patil, truy cập vào tháng 8 23, 2025, [https://medium.com/@rashmipatil24/javascript-es6-features-a-beginners-guide-to-modern-javascript-5786113ca9eb](https://medium.com/@rashmipatil24/javascript-es6-features-a-beginners-guide-to-modern-javascript-5786113ca9eb)  
9. map() filter() and reduce() in JavaScript \- GeeksforGeeks, truy cập vào tháng 8 23, 2025, [https://www.geeksforgeeks.org/javascript/how-to-use-map-filter-and-reduce-in-javascript/](https://www.geeksforgeeks.org/javascript/how-to-use-map-filter-and-reduce-in-javascript/)  
10. How to Use Async/Await in JavaScript – Explained with Code Examples \- freeCodeCamp, truy cập vào tháng 8 23, 2025, [https://www.freecodecamp.org/news/javascript-async-await/](https://www.freecodecamp.org/news/javascript-async-await/)  
11. Mastering DOM Manipulation: 10 Essential Tips for Efficient and High-Performance Web Development \- DEV Community, truy cập vào tháng 8 23, 2025, [https://dev.to/wizdomtek/mastering-dom-manipulation-10-essential-tips-for-efficient-and-high-performance-web-development-3mke](https://dev.to/wizdomtek/mastering-dom-manipulation-10-essential-tips-for-efficient-and-high-performance-web-development-3mke)  
12. Hooks at a Glance \- React, truy cập vào tháng 8 23, 2025, [https://legacy.reactjs.org/docs/hooks-overview.html](https://legacy.reactjs.org/docs/hooks-overview.html)  
13. Using the Effect Hook \- React, truy cập vào tháng 8 23, 2025, [https://legacy.reactjs.org/docs/hooks-effect.html](https://legacy.reactjs.org/docs/hooks-effect.html)  
14. Understanding React Lifecycle Methods and Hooks \- Medium, truy cập vào tháng 8 23, 2025, [https://medium.com/@rashmipatil24/understanding-react-lifecycle-methods-and-hooks-28522be7d7e4](https://medium.com/@rashmipatil24/understanding-react-lifecycle-methods-and-hooks-28522be7d7e4)  
15. Exploring React Hooks: Simplifying State and Lifecycle in Functional Components, truy cập vào tháng 8 23, 2025, [https://dev.to/anshumanmahato/exploring-react-hooks-simplifying-state-and-lifecycle-in-functional-components-56ch](https://dev.to/anshumanmahato/exploring-react-hooks-simplifying-state-and-lifecycle-in-functional-components-56ch)  
16. Context API Vs. Redux \- GeeksforGeeks, truy cập vào tháng 8 23, 2025, [https://www.geeksforgeeks.org/blogs/context-api-vs-redux-api/](https://www.geeksforgeeks.org/blogs/context-api-vs-redux-api/)  
17. React Context vs React Redux, when should I use each one? \[closed\] \- Stack Overflow, truy cập vào tháng 8 23, 2025, [https://stackoverflow.com/questions/49568073/react-context-vs-react-redux-when-should-i-use-each-one](https://stackoverflow.com/questions/49568073/react-context-vs-react-redux-when-should-i-use-each-one)  
18. Next.js application with app router | Uniform DXP documentation, truy cập vào tháng 8 23, 2025, [https://docs.uniform.app/docs/learn/tutorials/nextjs-app-router](https://docs.uniform.app/docs/learn/tutorials/nextjs-app-router)  
19. Getting Started: Server and Client Components \- Next.js, truy cập vào tháng 8 23, 2025, [https://nextjs.org/docs/app/getting-started/server-and-client-components](https://nextjs.org/docs/app/getting-started/server-and-client-components)  
20. Fetching Data \- App Router \- Next.js, truy cập vào tháng 8 23, 2025, [https://nextjs.org/learn/dashboard-app/fetching-data](https://nextjs.org/learn/dashboard-app/fetching-data)  
21. Server and Client Components \- React Foundations \- Next.js, truy cập vào tháng 8 23, 2025, [https://nextjs.org/learn/react-foundations/server-and-client-components](https://nextjs.org/learn/react-foundations/server-and-client-components)  
22. Getting Started: Updating Data \- Next.js, truy cập vào tháng 8 23, 2025, [https://nextjs.org/docs/app/getting-started/updating-data](https://nextjs.org/docs/app/getting-started/updating-data)  
23. Functions: Server Actions \- Next.js, truy cập vào tháng 8 23, 2025, [https://nextjs.org/docs/13/app/api-reference/functions/server-actions](https://nextjs.org/docs/13/app/api-reference/functions/server-actions)  
24. Image Optimization \- Next.js, truy cập vào tháng 8 23, 2025, [https://nextjs.org/docs/14/app/building-your-application/optimizing/images](https://nextjs.org/docs/14/app/building-your-application/optimizing/images)  
25. How to optimize images with the Next.js Image Component \- Contentful, truy cập vào tháng 8 23, 2025, [https://www.contentful.com/blog/nextjs-image-component/](https://www.contentful.com/blog/nextjs-image-component/)  
26. Mastering Route Handlers in Next.js: Simplified API Creation | by Prateek Badjatya | Medium, truy cập vào tháng 8 23, 2025, [https://prateekbadjatya.medium.com/mastering-route-handlers-in-next-js-simplified-api-creation-a12a0f739f8c](https://prateekbadjatya.medium.com/mastering-route-handlers-in-next-js-simplified-api-creation-a12a0f739f8c)  
27. Tailwind CSS: Utility-First Styling for Rapid UI Development \- GeeksforGeeks, truy cập vào tháng 8 23, 2025, [https://www.geeksforgeeks.org/blogs/tailwind-css-utility-first-styling-for-rapid-ui-development/](https://www.geeksforgeeks.org/blogs/tailwind-css-utility-first-styling-for-rapid-ui-development/)  
28. A Beginner's Guide to Utility Classes and Basic Usage \- TailGrids, truy cập vào tháng 8 23, 2025, [https://tailgrids.com/blog/tailwind-css-basic-usage-and-utility-classes](https://tailgrids.com/blog/tailwind-css-basic-usage-and-utility-classes)  
29. Tailwind CSS \- Utility-First Fundamentals \- Tutorialspoint, truy cập vào tháng 8 23, 2025, [https://www.tutorialspoint.com/tailwind\_css/tailwind\_css\_utility\_first\_fundamentals.htm](https://www.tutorialspoint.com/tailwind_css/tailwind_css_utility_first_fundamentals.htm)  
30. Styling with utility classes \- Core concepts \- Tailwind CSS, truy cập vào tháng 8 23, 2025, [https://tailwindcss.com/docs/styling-with-utility-classes](https://tailwindcss.com/docs/styling-with-utility-classes)  
31. Shadcn UI adoption guide: Overview, examples, and alternatives \- LogRocket Blog, truy cập vào tháng 8 23, 2025, [https://blog.logrocket.com/shadcn-ui-adoption-guide/](https://blog.logrocket.com/shadcn-ui-adoption-guide/)  
32. What is Shadcn UI and why you should use it?, truy cập vào tháng 8 23, 2025, [https://peerlist.io/blog/engineering/what-is-shadcn-and-why-you-should-use-it](https://peerlist.io/blog/engineering/what-is-shadcn-and-why-you-should-use-it)  
33. Shadcn-ui: What is it, and why do you care? \- WorkOS, truy cập vào tháng 8 23, 2025, [https://workos.com/blog/shadcn-ui](https://workos.com/blog/shadcn-ui)  
34. Introduction \- Shadcn UI, truy cập vào tháng 8 23, 2025, [https://ui.shadcn.com/docs](https://ui.shadcn.com/docs)  
35. Zustand: Introduction, truy cập vào tháng 8 23, 2025, [https://zustand.docs.pmnd.rs/](https://zustand.docs.pmnd.rs/)  
36. Managing React state with Zustand | by Dzmitry Ihnatovich \- Medium, truy cập vào tháng 8 23, 2025, [https://medium.com/@ignatovich.dm/managing-react-state-with-zustand-4e4d6bb50722](https://medium.com/@ignatovich.dm/managing-react-state-with-zustand-4e4d6bb50722)  
37. pmndrs/zustand: Bear necessities for state management in React \- GitHub, truy cập vào tháng 8 23, 2025, [https://github.com/pmndrs/zustand](https://github.com/pmndrs/zustand)  
38. Learn Zustand state management tool \- Abeer Abdul Ahad's Blog, truy cập vào tháng 8 23, 2025, [https://abeer.hashnode.dev/the-simplest-zustand-tutorial](https://abeer.hashnode.dev/the-simplest-zustand-tutorial)  
39. React State Management — using Zustand | by Chikku George | Globant \- Medium, truy cập vào tháng 8 23, 2025, [https://medium.com/globant/react-state-management-b0c81e0cbbf3](https://medium.com/globant/react-state-management-b0c81e0cbbf3)  
40. Build a User Management App with Next.js | Supabase Docs, truy cập vào tháng 8 23, 2025, [https://supabase.com/docs/guides/getting-started/tutorials/with-nextjs](https://supabase.com/docs/guides/getting-started/tutorials/with-nextjs)  
41. Building a backend without the hassle: how Supabase makes developers' lives easier, truy cập vào tháng 8 23, 2025, [https://www.halo-lab.com/blog/building-fast-backends-in-supabase](https://www.halo-lab.com/blog/building-fast-backends-in-supabase)  
42. Getting Started | Supabase Docs, truy cập vào tháng 8 23, 2025, [https://supabase.com/docs/guides/getting-started](https://supabase.com/docs/guides/getting-started)  
43. Building a simple To-Do app with Supabase & Next.js | by Nhyl Bryle Ibañez | Medium, truy cập vào tháng 8 23, 2025, [https://medium.com/@nbryleibanez/building-a-simple-to-do-app-with-supabase-next-js-2984ce16926a](https://medium.com/@nbryleibanez/building-a-simple-to-do-app-with-supabase-next-js-2984ce16926a)  
44. How to Build a Fullstack App with Next.js, Prisma & MongoDB \- Corbado, truy cập vào tháng 8 23, 2025, [https://www.corbado.com/blog/nextjs-prisma](https://www.corbado.com/blog/nextjs-prisma)  
45. How to Build a Fullstack App with Next.js, Prisma, and Postgres \- Vercel, truy cập vào tháng 8 23, 2025, [https://vercel.com/guides/nextjs-prisma-postgres](https://vercel.com/guides/nextjs-prisma-postgres)  
46. Using Prisma ORM with Next.js 15, TypeScript, and PostgreSQL \- DEV Community, truy cập vào tháng 8 23, 2025, [https://dev.to/mihir\_bhadak/using-prisma-orm-with-nextjs-15-typescript-and-postgresql-2b96](https://dev.to/mihir_bhadak/using-prisma-orm-with-nextjs-15-typescript-and-postgresql-2b96)  
47. What is Prisma ? How to use it with NextJS? \- YasH, truy cập vào tháng 8 23, 2025, [https://yashpurani.medium.com/what-is-prisma-how-to-use-it-with-nextjs-9742f4103d9](https://yashpurani.medium.com/what-is-prisma-how-to-use-it-with-nextjs-9742f4103d9)  
48. Installation \- Auth.js, truy cập vào tháng 8 23, 2025, [https://authjs.dev/getting-started/installation](https://authjs.dev/getting-started/installation)