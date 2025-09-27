# Giai đoạn 2: Tư duy với React

## Giới thiệu

Giai đoạn này là một bước nhảy vọt về tư duy. Thay vì viết các kịch bản tuần tự để thao tác trang web, bạn sẽ học cách xây dựng giao diện người dùng (UI) như một cây gồm các khối độc lập, có thể tái sử dụng và có trạng thái riêng, được gọi là **components**.

"Tư duy với React" là nghệ thuật phân rã một giao diện phức tạp thành những mảnh ghép nhỏ, dễ quản lý. Đây là lúc bạn thực sự bắt đầu xây dựng các ứng dụng web hiện đại.

## Danh sách Knowledge Points

### JSX và Components
- [[JSX syntax và rules]]
- [[React components là functions]]
- [[Component composition]]
- [[Props passing và validation]]
- [[Props children pattern]]
- [[Conditional rendering trong JSX]]
- [[Lists và keys trong React]]

### State Management
- [[useState Hook cơ bản]]
- [[State immutability rules]]
- [[Multiple state variables]]
- [[State batching và updates]]
- [[Lifting state up pattern]]
- [[State vs Props differences]]

### React Hooks
- [[useEffect Hook fundamentals]]
- [[useEffect dependency array]]
- [[useEffect cleanup functions]]
- [[Custom hooks creation]]
- [[useContext Hook]]
- [[useReducer cho complex state]]
- [[useMemo và useCallback optimization]]

### Event Handling
- [[Event handlers trong React]]
- [[Synthetic events]]
- [[Event delegation trong React]]
- [[Form handling patterns]]
- [[Controlled vs Uncontrolled components]]
- [[Form validation strategies]]

### Component Lifecycle
- [[Component mounting phase]]
- [[Component updating phase]]
- [[Component unmounting phase]]
- [[Effect cleanup patterns]]
- [[Dependency array best practices]]

### Context API
- [[React Context creation]]
- [[Provider và Consumer pattern]]
- [[useContext Hook usage]]
- [[Context value optimization]]
- [[When to use Context vs Props]]
- [[Context performance considerations]]

### Performance Optimization
- [[React.memo cho component memoization]]
- [[useMemo cho expensive calculations]]
- [[useCallback cho function memoization]]
- [[Key prop optimization]]
- [[Virtual DOM concepts]]
- [[Re-render optimization strategies]]

### Error Handling
- [[Error boundaries]]
- [[Try-catch trong async operations]]
- [[Error handling best practices]]
- [[Development vs Production errors]]

### React Development Tools
- [[React Developer Tools]]
- [[Component tree inspection]]
- [[Props và state debugging]]
- [[Performance profiling]]

## Mục tiêu cần đạt được

Nắm vững tư duy component-based, có khả năng phân rã một UI phức tạp thành các component nhỏ, tái sử dụng được, quản lý state và side effects một cách hiệu quả, và xây dựng được một ứng dụng trang đơn (Single-Page Application - SPA) hoàn chỉnh.

## Dự án thực hành gợi ý

Xây dựng một ứng dụng ghi chú (Note-taking app) hoặc To-do list. Ứng dụng phải cho phép người dùng thêm, sửa, xóa và đánh dấu hoàn thành các mục. 

Dự án này sẽ củng cố toàn bộ kiến thức về Components, Props, State (useState), Side Effects (useEffect để lưu dữ liệu vào localStorage của trình duyệt), và xử lý form.