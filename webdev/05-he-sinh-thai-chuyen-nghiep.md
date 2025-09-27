# Giai đoạn 4: Mở rộng Hệ sinh thái Chuyên nghiệp

## Giới thiệu

Sau khi đã làm chủ được Next.js, giai đoạn cuối cùng là trang bị cho mình bộ công cụ hoàn chỉnh để xây dựng các ứng dụng ở cấp độ sản xuất. 

Giai đoạn này tập trung vào việc tích hợp các thư viện và dịch vụ hàng đầu trong hệ sinh thái phát triển web hiện đại. Việc thành thạo những công cụ này không chỉ giúp tăng tốc độ phát triển mà còn đảm bảo ứng dụng của bạn có khả năng mở rộng, dễ bảo trì và mang lại trải nghiệm người dùng tuyệt vời.

Đây là bước để chuyển từ một người biết Next.js thành một kỹ sư phần mềm full-stack thực thụ.

## Danh sách Knowledge Points

### Styling với Tailwind CSS
- [[Utility-first CSS philosophy]]
- [[Tailwind CSS installation và setup]]
- [[Core utility classes]]
- [[Responsive design với Tailwind]]
- [[Tailwind color system]]
- [[Spacing và sizing utilities]]
- [[Layout utilities (flex, grid)]]
- [[Typography utilities]]
- [[Tailwind component patterns]]
- [[Custom utility classes]]
- [[Tailwind configuration file]]
- [[Dark mode với Tailwind]]
- [[Tailwind IntelliSense]]

### UI Library với Shadcn/ui
- [[Shadcn/ui architecture]]
- [[Installation và setup]]
- [[Component copying vs installation]]
- [[Radix UI fundamentals]]
- [[Customizing Shadcn components]]
- [[Theme configuration]]
- [[Component composition patterns]]
- [[Accessibility với Shadcn/ui]]
- [[Form components]]
- [[Navigation components]]
- [[Feedback components]]
- [[Data display components]]

### State Management với Zustand
- [[Zustand store creation]]
- [[State và actions definition]]
- [[Store subscription patterns]]
- [[Zustand vs Context API]]
- [[Zustand middleware]]
- [[Persistence với Zustand]]
- [[Zustand DevTools]]
- [[Store composition]]
- [[Async actions với Zustand]]
- [[TypeScript với Zustand]]

### Database với Supabase
- [[Supabase project setup]]
- [[PostgreSQL với Supabase]]
- [[Row Level Security (RLS)]]
- [[Supabase client configuration]]
- [[Database schema design]]
- [[Real-time subscriptions]]
- [[Supabase Storage]]
- [[Edge Functions]]
- [[Supabase CLI usage]]

### ORM với Prisma
- [[Prisma schema definition]]
- [[Database migrations]]
- [[Prisma Client generation]]
- [[CRUD operations với Prisma]]
- [[Relations và joins]]
- [[Prisma with Next.js integration]]
- [[Database seeding]]
- [[Prisma Studio]]
- [[Type safety với Prisma]]
- [[Performance optimization]]

### Authentication với Auth.js
- [[Auth.js setup với Next.js]]
- [[Providers configuration]]
- [[Session management]]
- [[Middleware protection]]
- [[Custom signin pages]]
- [[Database adapters]]
- [[JWT vs Database sessions]]
- [[Role-based access control]]
- [[OAuth providers setup]]
- [[Email provider configuration]]

### Deployment với Vercel
- [[Vercel project setup]]
- [[Git integration]]
- [[Environment variables]]
- [[Domain configuration]]
- [[Build optimization]]
- [[Serverless functions]]
- [[Edge functions]]
- [[Analytics với Vercel]]
- [[Preview deployments]]
- [[Production deployment strategies]]

### Advanced Topics
- [[TypeScript với Next.js]]
- [[Testing strategies]]
- [[Performance monitoring]]
- [[SEO optimization]]
- [[Security best practices]]
- [[Error tracking]]
- [[Monitoring và logging]]
- [[CI/CD pipelines]]

## Mục tiêu cần đạt được

Hoàn thiện bộ kỹ năng để xây dựng và triển khai một sản phẩm web full-stack hoàn chỉnh, an toàn, có khả năng mở rộng và sẵn sàng cho môi trường production.

## Dự án thực hành gợi ý

Phát triển một ứng dụng E-commerce nhỏ với các tính năng:

1. **Xác thực:** Người dùng có thể đăng ký và đăng nhập bằng email/mật khẩu hoặc tài khoản Google (sử dụng Auth.js).

2. **Hiển thị sản phẩm:** Dữ liệu sản phẩm được lưu trữ trong cơ sở dữ liệu Supabase và được truy vấn bằng Prisma trong các Server Components.

3. **Giao diện:** Giao diện người dùng được xây dựng bằng Next.js, Tailwind CSS, và các component có sẵn từ Shadcn/ui.

4. **Quản lý giỏ hàng:** Trạng thái giỏ hàng phía client được quản lý bằng Zustand.

5. **Triển khai:** Ứng dụng cuối cùng sẽ được triển khai lên Vercel để mọi người có thể truy cập.