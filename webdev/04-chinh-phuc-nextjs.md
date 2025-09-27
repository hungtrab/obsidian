# Giai đoạn 3: Chinh phục Next.js (App Router)

## Giới thiệu

Đây là giai đoạn cốt lõi của lộ trình, nơi bạn chuyển từ việc xây dựng các ứng dụng chỉ chạy phía client (SPA) sang các ứng dụng full-stack, hiệu năng cao với Next.js.

Giai đoạn này tập trung hoàn toàn vào **App Router**, mô hình mới và mạnh mẽ được giới thiệu từ Next.js 13, đại diện cho một sự thay đổi mô hình phát triển web. Thay vì mặc định mọi thứ diễn ra trên trình duyệt, App Router đưa logic trở lại máy chủ, kết hợp những ưu điểm tốt nhất của render phía máy chủ truyền thống và sự tương tác phong phú của các ứng dụng hiện đại.

## Danh sách Knowledge Points

### Routing System
- [[File-based routing với App Router]]
- [[page.tsx file convention]]
- [[layout.tsx file convention]]
- [[Dynamic routes với [slug]]]
- [[Route groups với (folder)]]
- [[Nested routes và layouts]]
- [[Route parameters và searchParams]]
- [[Parallel routes với @folder]]
- [[Intercepting routes với (.)folder]]

### Rendering Strategies
- [[Server Components vs Client Components]]
- [[use client directive]]
- [[Server-side rendering (SSR)]]
- [[Static Site Generation (SSG)]]
- [[Incremental Static Regeneration (ISR)]]
- [[Hybrid rendering strategies]]

### Server Components
- [[Server Components fundamentals]]
- [[Async Server Components]]
- [[Server Component benefits]]
- [[Server vs Client Component boundaries]]
- [[Passing data từ Server đến Client Components]]

### Client Components
- [[When to use Client Components]]
- [[Hydration process]]
- [[Interactive features với Client Components]]
- [[State management trong Client Components]]

### Data Fetching
- [[Fetch API trong Server Components]]
- [[Cache strategies với fetch]]
- [[revalidate options]]
- [[no-store cache option]]
- [[Data fetching patterns]]
- [[Loading states và error handling]]
- [[Parallel data fetching]]
- [[Sequential data fetching]]

### Server Actions
- [[Server Actions fundamentals]]
- [[use server directive]]
- [[Form handling với Server Actions]]
- [[revalidatePath và revalidateTag]]
- [[Error handling trong Server Actions]]
- [[Progressive enhancement]]
- [[Server Actions security]]

### Image Optimization
- [[Next.js Image component]]
- [[Image sizing và responsive images]]
- [[Image loading strategies]]
- [[Image formats optimization]]
- [[Lazy loading images]]
- [[Image placeholder strategies]]

### Font Optimization
- [[next/font fundamentals]]
- [[Google Fonts optimization]]
- [[Local fonts loading]]
- [[Font display strategies]]
- [[Font loading performance]]

### Navigation
- [[Link component]]
- [[useRouter Hook]]
- [[usePathname Hook]]
- [[useSearchParams Hook]]
- [[Programmatic navigation]]
- [[Prefetching strategies]]

### Route Handlers
- [[API Routes với route.ts]]
- [[HTTP methods (GET, POST, PUT, DELETE)]]
- [[Request và Response objects]]
- [[Middleware trong Route Handlers]]
- [[API Route authentication]]
- [[Error handling trong API Routes]]

### Metadata và SEO
- [[Static metadata]]
- [[Dynamic metadata]]
- [[generateMetadata function]]
- [[Open Graph metadata]]
- [[Twitter Card metadata]]
- [[Structured data]]

### Loading UI
- [[loading.tsx files]]
- [[Suspense boundaries]]
- [[Streaming với React Suspense]]
- [[Loading skeletons]]
- [[Progressive loading]]

### Error Handling
- [[error.tsx files]]
- [[Global error boundaries]]
- [[Not found pages]]
- [[Error recovery strategies]]

### Performance Optimization
- [[Bundle analysis]]
- [[Code splitting strategies]]
- [[Dynamic imports]]
- [[Route-based code splitting]]
- [[Performance monitoring]]

## Mục tiêu cần đạt được

Có khả năng xây dựng một ứng dụng web full-stack, hiệu năng cao, được tối ưu cho SEO bằng cách tận dụng sức mạnh của Server Components, Server Actions và các tính năng tối ưu hóa tích hợp của Next.js.

## Dự án thực hành gợi ý

Xây dựng một trang blog hoàn chỉnh. Trang chủ (Server Component) sẽ fetch và hiển thị danh sách các bài viết. Khi click vào một bài viết, người dùng sẽ được điều hướng đến một trang chi tiết (dynamic route) cũng được render trên server. 

Sẽ có một trang liên hệ với một form sử dụng Server Actions để gửi dữ liệu. Tất cả hình ảnh sẽ được tối ưu hóa bằng component Image.