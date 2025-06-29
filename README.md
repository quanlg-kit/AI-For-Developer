# Hướng dẫn chi tiết 8 Module khóa học "Ứng dụng AI cho Developers"

Dựa trên nội dung khóa học đã tổng hợp, tôi sẽ trình bày chi tiết từng module với hướng dẫn step-by-step và ví dụ thực hành.

## **Module 01: Basic LLMs & Prompts, LM Studio**

### **Bài 1.1: Hiểu về LLMs và cơ chế hoạt động**

**Khái niệm cơ bản:**
Large Language Model (LLM) là thuật toán deep learning mạnh mẽ với khả năng xử lý nhiều tác vụ trong natural language processing[1]. LLM được xây dựng trên kiến trúc Transformer và trải qua quá trình đào tạo chuyên sâu với các bộ dữ liệu khổng lồ[1].

**Cơ chế hoạt động của LLM gồm 4 bước chính:**

1. **Tokenization**: Chia văn bản thành các token (đơn vị nhỏ hơn)[2]
2. **Embedding**: Chuyển token thành vector số học[2]
3. **Processing**: Xử lý qua các lớp Transformer với cơ chế Self-Attention[2]
4. **Generation**: Dự đoán token tiếp theo và sinh văn bản[2]

**Ví dụ thực tế:**
Khi bạn hỏi "Nên ăn gì vào buổi sáng nhỉ?", LLM sẽ:
- Tokenize thành: ["Nên", "ăn", "gì", "vào", "buổi", "sáng", "nhỉ", "?"]
- Dự đoán từng từ tiếp theo: "Bạn" → "có" → "nhiều" → "lựa chọn"...[3]

### **Bài 1.2: Tokenizer và cách hoạt động**

**Tokenizer là gì:**
Tokenizer là thành phần cốt lõi chuyển đổi văn bản thành dữ liệu số mà mô hình có thể xử lý[4]. Có 3 loại tokenizer chính:

1. **Word-based tokenizer**: Chia theo từ[4]
2. **Character-based tokenizer**: Chia theo ký tự
3. **Subword tokenizer**: Chia theo các phần nhỏ của từ (phổ biến nhất)[4]

**Ảnh hưởng của Tokenization:**
- **Context Window Limit**: Mỗi LLM có giới hạn số token có thể xử lý[5]
- **Hiệu suất**: Cách tokenize ảnh hưởng đến khả năng hiểu ngữ cảnh[5]

**Thực hành:**
```python
# Ví dụ với transformers library
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Tôi thích lập trình AI"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
print(f"Token IDs: {tokenizer.convert_tokens_to_ids(tokens)}")
```

### **Bài 1.3: Các thông số chính trong LLM**

**Các thông số quan trọng:**

1. **Temperature** (0.0 - 2.0): Kiểm soát độ sáng tạo
   - 0.0: Câu trả lời cố định, dự đoán được
   - 1.0: Cân bằng giữa sáng tạo và logic
   - 2.0: Rất sáng tạo, có thể không logic[6]

2. **Top-p** (0.0 - 1.0): Lọc các từ có xác suất thấp
   - 0.9: Chọn từ trong 90% xác suất cao nhất[6]

3. **Max tokens**: Giới hạn độ dài phản hồi

4. **Context window**: Số token tối đa mô hình có thể "nhớ"

**Ví dụ cài đặt:**
```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 1000,
  "context_window": 4096
}
```

### **Bài 1.4: Kỹ thuật Prompt Engineering cơ bản**

**Các kỹ thuật prompt chính:**

1. **Zero-shot Prompting**: Không cung cấp ví dụ[7]
```
Phân loại bài đánh giá này là TÍCH CỰC, TRUNG LẬP hoặc TIÊU CỰC: 
"Sản phẩm này rất tốt và chất lượng"
```

2. **Few-shot Prompting**: Cung cấp vài ví dụ[7]
```
Dịch từ tiếng Anh sang tiếng Việt:
- Hello → Xin chào
- Thank you → Cảm ơn
- Good morning → ___
```

3. **Chain-of-Thought (CoT)**: Hướng dẫn suy luận từng bước[6]
```
Giải bài toán theo từng bước:
Bước 1: Xác định vấn đề
Bước 2: Liệt kê các phương pháp
Bước 3: Chọn phương pháp tối ưu
Bước 4: Triển khai
```

### **Bài 1.5: LM Studio - Cài đặt và sử dụng**

**Cài đặt LM Studio:**

**Bước 1**: Tải về từ https://lmstudio.ai[8]

**Bước 2**: Yêu cầu hệ thống[8]
- RAM: 8GB (khuyến nghị 16GB+)
- Dung lượng: 10GB+
- GPU: Không bắt buộc nhưng khuyến khích 4GB+ VRAM

**Bước 3**: Cài đặt theo hướng dẫn

**Sử dụng LM Studio:**

**Bước 1**: Tải model
- Mở LM Studio → chọn "Discover"
- Tìm kiếm model (khuyến nghị bắt đầu với 7B-8B parameters)[8]
- Download model phù hợp với cấu hình:
  - CPU/RAM 8GB: 7B-8B (2-4 bit quantized)
  - GPU 6-8GB VRAM: 7B-13B (4-bit quantized)
  - GPU 16GB+ VRAM: 13B+ (8-bit quantized)[8]

**Bước 2**: Chạy model
- Chọn model đã tải → Load
- Điều chỉnh thông số (temperature, top-p)
- Bắt đầu chat

**Bước 3**: Sử dụng API
```python
import requests

response = requests.post('http://localhost:1234/v1/chat/completions', 
    json={
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7
    })
print(response.json())
```

## **Module 02: Tìm hiểu Windsurf/Cursor/Cline**

### **Bài 2.1: So sánh các công cụ AI Coding**

**Cursor - AI Pair Programmer:**
- Giá: $20/tháng[9]
- Điểm mạnh: Project-level awareness, nhiều models hỗ trợ[9]
- Phù hợp: Developer muốn AI assistant thông minh như Copilot nhưng mạnh hơn[10]

**Windsurf - UI/UX tốt:**
- Giá: $15/tháng[9]
- Điểm mạnh: Giao diện đẹp, animation mượt, thinking mode tiết kiệm[9]
- Phù hợp: Xây dựng apps, internal tools với UI patterns lặp lại[10]

**Cline - Multi-agent:**
- Điểm mạnh: Nhiều AI agents làm việc cùng nhau, debugging mạnh[10]
- Phù hợp: Teams cần debugging nhanh, onboarding junior devs[10]

### **Bài 2.2: Demo thực hành - Xây dựng Chatbot App**

**Bước 1**: Khởi tạo project
```bash
mkdir ai-chatbot
cd ai-chatbot
npm init -y
npm install express socket.io
```

**Bước 2**: Thiết lập cơ bản với Cursor
- Mở Cursor, tạo file `server.js`
- Sử dụng Cursor Chat: "Tạo một Express server với Socket.io cho chatbot"

**Cursor sẽ generate:**
```javascript
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

app.use(express.static('public'));

io.on('connection', (socket) => {
    console.log('User connected');
    
    socket.on('message', (msg) => {
        // Xử lý tin nhắn
        socket.emit('response', `Bot: ${msg}`);
    });
});

server.listen(3000, () => {
    console.log('Server running on port 3000');
});
```

**Bước 3**: Tạo frontend với Windsurf
- Prompt: "Tạo HTML interface cho chatbot với design hiện đại"

**Windsurf sẽ generate:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>AI Chatbot</title>
    <style>
        .chat-container { /* Windsurf tạo CSS đẹp */ }
        .message { /* Animation mượt mà */ }
    </style>
</head>
<body>
    <div class="chat-container">
        <div id="messages"></div>
        <input type="text" id="messageInput" placeholder="Nhập tin nhắn...">
    </div>
    <script src="/socket.io/socket.io.js"></script>
    <script>
        const socket = io();
        // Logic chat
    </script>
</body>
</html>
```

**Bước 4**: Tích hợp AI với Cline
- Cline sẽ tự động phát hiện và đề xuất tích hợp OpenAI API
- Tạo agent xử lý multi-context conversations

## **Module 03: Nghiên cứu giải pháp với AI Tools**

### **Bài 3.1: Sử dụng Manus, Perplexity, Genspark**

**Manus - AI Research Assistant:**
- Chức năng: Nghiên cứu sâu, phân tích tài liệu
- Thực hành: "Nghiên cứu về microservices architecture cho e-commerce"

**Perplexity - Real-time Search:**
- Chức năng: Tìm kiếm thông tin real-time với citations
- Thực hành: "Latest React 18 best practices 2025"

**Genspark - Technical Analysis:**
- Chức năng: Phân tích kỹ thuật chuyên sâu
- Thực hành: "Compare database performance: PostgreSQL vs MongoDB"

### **Bài 3.2: Quy trình nghiên cứu giải pháp**

**Bước 1**: Xác định vấn đề
```
Project: E-commerce platform
Problem: High latency in product search
Requirements: <1s response time, 10M+ products
```

**Bước 2**: Nghiên cứu với Perplexity
```
Prompt: "Best practices for scaling product search in e-commerce 2025. 
Include: Elasticsearch vs Algolia vs custom solutions, 
real-world case studies, performance benchmarks"
```

**Bước 3**: Phân tích sâu với Manus
```
Upload documents: Elasticsearch docs, case studies
Query: "Compare implementation complexity, cost, and performance 
for 10M product catalog"
```

**Bước 4**: Tạo comparison matrix
| Solution | Performance | Cost | Complexity | Scalability |
|----------|-------------|------|------------|-------------|
| Elasticsearch | 8/10 | 6/10 | 7/10 | 9/10 |
| Algolia | 9/10 | 4/10 | 9/10 | 8/10 |
| Custom | 7/10 | 8/10 | 4/10 | 6/10 |

### **Bài 3.3: Tạo slide thuyết trình với AI**

**Template prompt cho slide generation:**
```
Tạo slide thuyết trình cho giải pháp product search:

Slide 1: Problem Statement
- Current issues: 3-5s search time
- User impact: 40% bounce rate
- Business impact: $2M revenue loss/year

Slide 2: Solution Comparison
[Include comparison matrix]

Slide 3: Recommended Solution
- Elasticsearch với Redis cache
- Expected improvement: <1s response
- Implementation timeline: 3 months

Slide 4: Technical Architecture
[Diagram của solution]

Slide 5: Implementation Plan
- Phase 1: Setup & Migration (1 month)
- Phase 2: Optimization (1 month) 
- Phase 3: Monitoring & Tuning (1 month)
```

## **Module 04: Context Optimization**

### **Bài 4.1: Hiểu về Context trong AI**

**Context là gì:**
Context là tất cả thông tin mà AI model có thể "nhìn thấy" và sử dụng để đưa ra phản hồi. Bao gồm:
- Prompt hiện tại
- Lịch sử cuộc trò chuyện
- Code files được reference
- Documentation được include[9]

**Tại sao Context quan trọng:**
- Quyết định độ chính xác của output
- Ảnh hưởng đến consistency
- Giới hạn bởi context window (tokens)[5]

### **Bài 4.2: Context Window và Limitations**

**Context Window Sizes (2025):**
- GPT-4: 128K tokens
- Claude 3.5 Sonnet: 200K tokens
- Cursor Agent: 60-120K tokens[9]

**Vấn đề khi vượt Context Window:**
- AI "quên" thông tin cũ
- Inconsistent behavior
- Performance degradation[5]

### **Bài 4.3: Tối ưu Context trên Cursor**

**Cursor Context Features:**

1. **@Files**: Reference specific files
```
@package.json @src/components/Button.tsx
Refactor this Button component to use TypeScript generics
```

2. **@Folders**: Include entire directories
```
@src/utils 
Review all utility functions for performance issues
```

3. **@Web**: Include web search results
```
@web latest React 19 features
Update our components to use new React 19 APIs
```

4. **@Docs**: Reference documentation
```
@docs next.js app router
Convert this pages router to app router
```

**Best Practices:**
- Chỉ include relevant files
- Sử dụng .cursorignore để exclude files không cần thiết
- Prioritize recent và related code

### **Bài 4.4: Tối ưu Context trên Windsurf**

**Windsurf Context Management:**

1. **Smart Context Selection**: Windsurf tự động chọn relevant files
2. **Context Compression**: Tự động nén context khi gần limit
3. **Thinking Mode**: Giảm token consumption[9]

**Configuration:**
```json
// .windsurf/config.json
{
  "context": {
    "maxFiles": 20,
    "autoInclude": ["*.ts", "*.tsx", "package.json"],
    "exclude": ["node_modules", "dist", ".git"]
  }
}
```

### **Bài 4.5: Thực hành tối ưu Context**

**Scenario**: Debugging một React app với 50+ components

**Trước khi tối ưu:**
```
@src/ 
This app has a memory leak, please debug
```
Problem: Too much context, AI overwhelmed

**Sau khi tối ưu:**
```
@src/hooks/useMemoryHook.ts @src/components/DataList.tsx @package.json
Memory leak occurs when DataList unmounts. Focus on useMemoryHook 
and DataList component. Check for:
1. Event listeners cleanup
2. Timer/interval cleanup
3. Subscription cleanup
```

**Kết quả**: AI focused, faster và accurate hơn

## **Module 05: Hexagonal Architecture với Cursor AI**

### **Bài 5.1: Giới thiệu Hexagonal Architecture**

**Hexagonal Architecture là gì:**
Hexagonal Architecture (hay Ports and Adapters) là mẫu kiến trúc tách biệt business logic khỏi external concerns[11]. Ứng dụng được đặt ở trung tâm, giao tiếp với bên ngoài qua ports/adapters[11].

**Lợi ích:**
- Dễ test và maintain[11]
- Không phụ thuộc vào framework/library[11]
- Có thể implement business logic trước khi chọn technology stack[11]

**Cấu trúc:**
```
Domain/
├── models/          # Business entities
├── repositories/    # Data access interfaces  
└── services/        # Business logic

Infrastructure/
├── http/           # API controllers
├── database/       # Database implementations
└── external/       # External service integrations
```

### **Bài 5.2: Thiết lập project với Cursor AI**

**Bước 1**: Tạo Cursor Rules cho Hexagonal Architecture
```
// .cursor-rules
Project follows Hexagonal Architecture principles:

DOMAIN LAYER:
- models/: Pure business entities, no external dependencies
- repositories/: Interfaces only, no implementations
- services/: Business logic, depends only on domain

INFRASTRUCTURE LAYER:  
- http/: Express controllers, DTOs
- database/: Repository implementations
- external/: API clients, third-party integrations

RULES:
1. Domain never depends on Infrastructure
2. Infrastructure can depend on Domain
3. Use dependency injection
4. All external calls through interfaces
5. Business logic in services only
```

**Bước 2**: Generate project structure
```
Prompt cho Cursor:
"Create a Node.js e-commerce project following Hexagonal Architecture.
Include: User management, Product catalog, Order processing.
Generate folder structure, base classes, and dependency injection setup."
```

### **Bài 5.3: Domain Layer Implementation**

**Models (Entities):**
```typescript
// domain/models/User.ts
export class User {
  constructor(
    public readonly id: string,
    public readonly email: string,
    public readonly name: string,
    private _isActive: boolean = true
  ) {}

  activate(): void {
    this._isActive = true;
  }

  deactivate(): void {
    this._isActive = false;
  }

  get isActive(): boolean {
    return this._isActive;
  }
}
```

**Repository Interfaces:**
```typescript
// domain/repositories/UserRepository.ts
import { User } from '../models/User';

export interface UserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  save(user: User): Promise<void>;
  delete(id: string): Promise<void>;
}
```

**Domain Services:**
```typescript
// domain/services/UserService.ts
import { User } from '../models/User';
import { UserRepository } from '../repositories/UserRepository';

export class UserService {
  constructor(private userRepository: UserRepository) {}

  async createUser(email: string, name: string): Promise<User> {
    const existingUser = await this.userRepository.findByEmail(email);
    if (existingUser) {
      throw new Error('User already exists');
    }

    const user = new User(generateId(), email, name);
    await this.userRepository.save(user);
    return user;
  }

  async activateUser(id: string): Promise<void> {
    const user = await this.userRepository.findById(id);
    if (!user) {
      throw new Error('User not found');
    }

    user.activate();
    await this.userRepository.save(user);
  }
}
```

### **Bài 5.4: Infrastructure Layer Implementation**

**Database Repository Implementation:**
```typescript
// infrastructure/database/MongoUserRepository.ts
import { UserRepository } from '../../domain/repositories/UserRepository';
import { User } from '../../domain/models/User';

export class MongoUserRepository implements UserRepository {
  async findById(id: string): Promise<User | null> {
    // MongoDB implementation
    const userData = await UserModel.findById(id);
    return userData ? this.toDomain(userData) : null;
  }

  async save(user: User): Promise<void> {
    await UserModel.findByIdAndUpdate(
      user.id, 
      this.toPersistence(user), 
      { upsert: true }
    );
  }

  private toDomain(userData: any): User {
    return new User(userData._id, userData.email, userData.name, userData.isActive);
  }

  private toPersistence(user: User): any {
    return {
      _id: user.id,
      email: user.email,
      name: user.name,
      isActive: user.isActive
    };
  }
}
```

**HTTP Controllers:**
```typescript
// infrastructure/http/UserController.ts
import { Request, Response } from 'express';
import { UserService } from '../../domain/services/UserService';

export class UserController {
  constructor(private userService: UserService) {}

  async createUser(req: Request, res: Response): Promise<void> {
    try {
      const { email, name } = req.body;
      const user = await this.userService.createUser(email, name);
      res.status(201).json({ id: user.id, email: user.email, name: user.name });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async activateUser(req: Request, res: Response): Promise<void> {
    try {
      await this.userService.activateUser(req.params.id);
      res.status(200).json({ message: 'User activated' });
    } catch (error) {
      res.status(404).json({ error: error.message });
    }
  }
}
```

### **Bài 5.5: Dependency Injection Setup**

**DI Container:**
```typescript
// infrastructure/di/Container.ts
import { UserService } from '../../domain/services/UserService';
import { UserRepository } from '../../domain/repositories/UserRepository';
import { MongoUserRepository } from '../database/MongoUserRepository';
import { UserController } from '../http/UserController';

export class Container {
  private static instance: Container;
  private services = new Map();

  static getInstance(): Container {
    if (!Container.instance) {
      Container.instance = new Container();
    }
    return Container.instance;
  }

  register(): void {
    // Repositories
    this.services.set('UserRepository', new MongoUserRepository());
    
    // Services  
    this.services.set('UserService', new UserService(
      this.services.get('UserRepository')
    ));
    
    // Controllers
    this.services.set('UserController', new UserController(
      this.services.get('UserService')  
    ));
  }

  get<T>(key: string): T {
    return this.services.get(key);
  }
}
```

**Cursor prompt để generate test:**
```
"Generate unit tests for UserService following Hexagonal Architecture.
Mock the UserRepository interface. Test business logic only.
Include positive and negative test cases."
```

## **Module 06: Cursor/Windsurf Rules & Memory Bank**

### **Bài 6.1: Cursor Rules System**

**Cursor Rules là gì:**
Cursor Rules là cách để hướng dẫn AI về coding style, patterns, và conventions của project[9]. Rules được stored trong `.cursor-rules` file.

**Cấp độ Rules:**
1. **Global Rules**: Áp dụng cho tất cả projects
2. **Project Rules**: Specific cho từng project  
3. **File Rules**: Specific cho từng file type

### **Bài 6.2: Thiết lập Rules cho Hexagonal Architecture**

**Base Rules:**
```
# .cursor-rules

## HEXAGONAL ARCHITECTURE RULES

### DOMAIN LAYER
- domain/ folder contains pure business logic
- No external dependencies in domain layer
- Models are immutable value objects
- Services contain business rules only
- Repository interfaces, no implementations

Example domain service:
```
export class OrderService {
  constructor(private orderRepo: OrderRepository) {}
  
  async createOrder(customerId: string, items: OrderItem[]): Promise<Order> {
    // Business validation
    if (items.length === 0) throw new Error('Order must have items');
    
    const order = new Order(generateId(), customerId, items);
    await this.orderRepo.save(order);
    return order;
  }
}
```

### INFRASTRUCTURE LAYER  
- infrastructure/ contains external concerns
- Database implementations
- HTTP controllers with DTOs
- External API clients

### DEPENDENCY INJECTION
- Use constructor injection
- Depend on interfaces, not implementations
- Register dependencies in DI container
```

**Framework-specific Rules:**
```
### EXPRESS.JS RULES
- Controllers handle HTTP concerns only
- Use DTOs for request/response
- Validate input in controllers
- Handle errors with middleware

### TYPESCRIPT RULES
- Strict typing enabled
- No any types
- Use interfaces for contracts
- Export types from index files

### TESTING RULES
- Unit tests for domain layer
- Integration tests for infrastructure
- Mock external dependencies
- Test business logic thoroughly
```

### **Bài 6.3: Windsurf Memory Bank**

**Memory Bank Features:**
- Lưu trữ patterns đã sử dụng
- Remember project context across sessions
- Auto-suggest based on history[9]

**Configuration:**
```json
// .windsurf/memory.json
{
  "patterns": {
    "hexagonal_service": {
      "template": "class {{ServiceName}} { constructor(private {{repoName}}: {{RepoInterface}}) {} }",
      "usage_count": 15,
      "last_used": "2025-01-15"
    },
    "dto_pattern": {
      "template": "export interface {{EntityName}}DTO { id: string; {{fields}} }",
      "usage_count": 8,
      "last_used": "2025-01-14"  
    }
  }
}
```

### **Bài 6.4: Advanced Rules Configuration**

**File Pattern Rules:**
```
# Pattern-specific rules

## When creating *.service.ts files:
- Follow domain service pattern
- Constructor inject repositories
- Return domain entities
- Throw domain exceptions
- No HTTP concerns

## When creating *.controller.ts files:  
- Handle HTTP requests only
- Validate input DTOs
- Call domain services
- Transform to response DTOs
- Handle errors appropriately

## When creating *.repository.ts files:
- Implement repository interface
- Handle data persistence
- Transform between domain and data models
- No business logic
```

**Auto-generation Rules:**
```
## Auto-generate patterns:

WHEN creating service:
1. Generate interface first
2. Generate implementation  
3. Generate unit tests
4. Register in DI container

WHEN creating entity:
1. Generate domain model
2. Generate DTO
3. Generate mapper functions
4. Generate repository interface
```

### **Bài 6.5: Thực hành với Rules**

**Scenario**: Tạo Product management module

**Bước 1**: Prompt với Rules
```
Using hexagonal architecture rules, create Product management:
- Product entity with validation
- ProductService with business rules  
- ProductRepository interface
- ProductController with DTOs
- Unit tests for business logic
```

**Cursor sẽ generate theo Rules:**

```typescript
// domain/models/Product.ts - Follows rules
export class Product {
  constructor(
    public readonly id: string,
    public readonly name: string,
    public readonly price: number,
    public readonly category: string
  ) {
    if (price < 0) throw new Error('Price cannot be negative');
    if (!name.trim()) throw new Error('Name is required');
  }
}

// domain/services/ProductService.ts - Follows service rules  
export class ProductService {
  constructor(private productRepo: ProductRepository) {}
  
  async createProduct(name: string, price: number, category: string): Promise<Product> {
    // Business validation as per rules
    const product = new Product(generateId(), name, price, category);
    await this.productRepo.save(product);
    return product;
  }
}
```

**Bước 2**: Verify Rules Compliance
Cursor tự động check và suggest fixes nếu vi phạm rules.

## **Module 07: MCP - Model Context Protocol**

### **Bài 7.1: Giới thiệu MCP**

**MCP là gì:**
Model Context Protocol là open standard được Anthropic giới thiệu tháng 11/2024 để chuẩn hóa cách AI models tích hợp với external tools và data sources[12][13]. MCP thay thế các custom integrations bằng một protocol thống nhất[13].

**Kiến trúc MCP:**
- **MCP Server**: Expose data/tools qua MCP protocol
- **MCP Client**: AI applications connect to MCP servers  
- **Transport**: JSON-RPC 2.0 over stdio, SSE, hoặc HTTP[12]

**Lợi ích:**
- Tăng scalability và maintainability[14]
- Giảm duplicate development effort[14] 
- Tránh vendor lock-in[14]
- Tăng tốc development process[14]

### **Bài 7.2: Context7 và Multi-Context Processing**

**Context7 Overview:**
Context7 là MCP server cho phép AI xử lý multiple contexts simultaneously, rất hữu ích cho complex requirements[14].

**Use Cases:**
- Analyze code + documentation + requirements cùng lúc
- Cross-reference multiple data sources
- Maintain context across different tools[14]

**Thực hành Setup:**
```bash
# Install Context7 MCP server
npm install -g context7-mcp-server

# Configuration
{
  "mcpServers": {
    "context7": {
      "command": "context7-mcp-server",
      "args": ["--config", "./context7.json"]
    }
  }
}
```

### **Bài 7.3: Figma to Code Integration**

**Figma to Code với MCP:**
MCP cho phép AI trực tiếp access Figma designs và convert thành code với độ chính xác ~60%[15].

**Setup Figma MCP:**
```bash
# Install Figma MCP server
npm install figma-mcp-server

# Config in Claude/Cursor
{
  "mcpServers": {
    "figma": {
      "command": "figma-mcp-server",
      "args": ["--token", "YOUR_FIGMA_TOKEN"]
    }
  }
}
```

**Workflow:**
1. **Design Analysis**: MCP server fetches Figma design
2. **Component Detection**: Identifies UI components và layout
3. **Code Generation**: Converts to React/Vue/HTML[15]
4. **Optimization**: Applies responsive patterns[15]

**Ví dụ sử dụng:**
```
Prompt: "Convert this Figma design to React components:
https://www.figma.com/file/abc123/ecommerce-dashboard

Requirements:
- Use TypeScript
- Tailwind CSS
- Responsive design  
- Component-based architecture"
```

**Output được tối ưu:**
```tsx
// Generated by Figma MCP
export interface DashboardProps {
  metrics: MetricData[];
  charts: ChartConfig[];
}

export const Dashboard: React.FC<DashboardProps> = ({ metrics, charts }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 p-6">
      {metrics.map(metric => (
        <MetricCard key={metric.id} {...metric} />
      ))}
      <div className="col-span-full">
        <ChartContainer charts={charts} />
      </div>
    </div>
  );
};
```

### **Bài 7.4: Database Integration với MCP**

**Database MCP Server:**
MCP cho phép AI interact trực tiếp với databases qua standardized interface[16][17].

**Multi-Database Support:**
```json
{
  "connections": [
    {
      "id": "postgres_main",
      "type": "postgres", 
      "host": "localhost",
      "port": 5432,
      "database": "ecommerce",
      "user": "dev_user",
      "password": "dev_pass"
    },
    {
      "id": "mysql_analytics", 
      "type": "mysql",
      "host": "analytics.company.com",
      "port": 3306,
      "database": "analytics",
      "user": "analyst",
      "password": "analyst_pass"
    }
  ]
}
```

**Auto-generated Tools:**
Mỗi database connection tự động tạo các tools:
- `query_postgres_main`: Execute SELECT queries
- `execute_postgres_main`: Run INSERT/UPDATE/DELETE
- `schema_postgres_main`: Explore database schema
- `transaction_postgres_main`: Manage transactions[16]

**Thực hành:**
```
Prompt: "Analyze user behavior from both databases:
1. Get user orders from postgres_main
2. Get user analytics from mysql_analytics  
3. Create comprehensive user profile report
4. Suggest personalization strategies"
```

AI sẽ:
1. Query user data từ PostgreSQL
2. Fetch analytics từ MySQL
3. Cross-reference data
4. Generate insights

### **Bài 7.5: Custom MCP Server Development**

**Tạo MCP Server cho GitHub Integration:**

**Bước 1**: Setup project
```bash
mkdir github-mcp-server
cd github-mcp-server
npm init -y
npm install @anthropic/sdk @octokit/rest
```

**Bước 2**: Implement MCP Server
```typescript
// src/server.ts
import { MCPServer } from '@anthropic/mcp-server';
import { Octokit } from '@octokit/rest';

export class GitHubMCPServer extends MCPServer {
  private octokit: Octokit;

  constructor(token: string) {
    super();
    this.octokit = new Octokit({ auth: token });
  }

  async initialize() {
    // Register tools
    this.registerTool({
      name: 'list_repositories',
      description: 'List user repositories',
      parameters: {
        type: 'object',
        properties: {
          username: { type: 'string' }
        }
      },
      handler: this.listRepositories.bind(this)
    });

    this.registerTool({
      name: 'analyze_codebase', 
      description: 'Analyze repository codebase',
      parameters: {
        type: 'object',
        properties: {
          owner: { type: 'string' },
          repo: { type: 'string' }
        }
      },
      handler: this.analyzeCodebase.bind(this)
    });
  }

  async listRepositories(params: { username: string }) {
    const { data } = await this.octokit.repos.listForUser({
      username: params.username
    });
    
    return data.map(repo => ({
      name: repo.name,
      description: repo.description,
      stars: repo.stargazers_count,
      language: repo.language
    }));
  }

  async analyzeCodebase(params: { owner: string; repo: string }) {
    // Get repository contents
    const { data: contents } = await this.octokit.repos.getContent({
      owner: params.owner,
      repo: params.repo,
      path: ''
    });

    // Analyze file structure, dependencies, etc.
    const analysis = await this.performCodeAnalysis(contents);
    return analysis;
  }
}
```

**Bước 3**: Register với Cursor/Claude
```json
{
  "mcpServers": {
    "github": {
      "command": "node",
      "args": ["./dist/server.js"],
      "env": {
        "GITHUB_TOKEN": "your_github_token"
      }
    }
  }
}
```

**Bước 4**: Sử dụng
```
Prompt: "Analyze my GitHub repositories and suggest:
1. Which repos need documentation updates
2. Which repos have security vulnerabilities  
3. Code quality improvements needed
4. Potential refactoring opportunities"
```

## **Module 08: Claude Task Master**

### **Bài 8.1: Giới thiệu Claude Task Master**

**Task Master là gì:**
Claude Task Master là AI-powered task management system được thiết kế để tích hợp với AI-driven code editors như Cursor AI, Windsurf, và Roo[18][19]. Nó sử dụng Anthropic API (Claude) để tự động hóa việc tạo, prioritize, và track tasks[18].

**Key Features:**
- Parse PRDs thành actionable tasks[18] 
- AI-powered task creation và prioritization[18]
- Integration với Cursor AI và AI-driven IDEs[18]
- Command-line interface cho task management[18]
- Automated task file generation[18]

### **Bài 8.2: Cài đặt và Configuration**

**Installation Options:**

**Option 1: MCP Integration (Recommended)**
```bash
# One-click install via MCP link
https://task-master-ai.com/install-mcp
```

**Option 2: Command Line**
```bash
# Global installation
npm install -g task-master-ai

# Or local installation
npm install task-master-ai
```

**API Keys Setup:**
Task Master hỗ trợ multiple AI providers[19]:
- Anthropic API key (Claude API) - Recommended
- OpenAI API key
- Google Gemini API key  
- Perplexity API key (cho research model)
- xAI API Key
- OpenRouter API Key
- Claude Code (không cần API key)[19]

**Configuration:**
```json
// .env file
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key

// task-master.config.json
{
  "models": {
    "main": "claude-3-5-sonnet-20241022",
    "research": "perplexity-llama-3.1-sonar-huge-128k-online", 
    "fallback": "gpt-4"
  },
  "project": {
    "rules": ["cursor", "windsurf", "typescript"],
    "framework": "next.js"
  }
}
```

### **Bài 8.3: Project Initialization**

**Initialize New Project:**
```bash
# Basic initialization
task-master init

# With specific rules
task-master init --rules cursor,windsurf,typescript,react

# Interactive setup
task-master init --interactive
```

**Project Setup Process:**
1. **Project Details**: Name, description, framework
2. **Rules Selection**: Coding standards, linting rules
3. **AI Models**: Main, research, fallback models
4. **Structure Generation**: Folder structure, base files
5. **Integration Setup**: Cursor/Windsurf configuration

**Generated Structure:**
```
my-project/
├── .task-master/
│   ├── config.json
│   ├── tasks.json
│   └── rules/
├── docs/
│   └── requirements.md
├── src/
├── tests/
└── .cursor-rules
```

### **Bài 8.4: PRD Parsing và Task Generation**

**PRD (Product Requirements Document) Parsing:**

**Ví dụ PRD:**
```markdown
# E-commerce Platform Requirements

## Overview
Build a modern e-commerce platform with user management, product catalog, and order processing.

## Features
1. User Authentication & Authorization
   - Email/password login
   - Social login (Google, Facebook)
   - Role-based access control

2. Product Management
   - Product CRUD operations
   - Category management
   - Inventory tracking
   - Image upload

3. Shopping Cart & Checkout
   - Add/remove items
   - Payment integration (Stripe)
   - Order confirmation

## Technical Requirements
- Frontend: Next.js with TypeScript
- Backend: Node.js with Express
- Database: PostgreSQL
- Authentication: NextAuth.js
```

**Parse PRD Command:**
```bash
task-master parse-prd requirements.md
```

**Generated Tasks:**
```json
{
  "tasks": [
    {
      "id": "001",
      "title": "Setup Next.js project with TypeScript",
      "description": "Initialize Next.js project with TypeScript configuration",
      "category": "setup",
      "priority": "high",
      "estimatedTime": "2 hours",
      "dependencies": [],
      "subtasks": [
        "Create Next.js project",
        "Configure TypeScript",
        "Setup ESLint and Prettier",
        "Configure path aliases"
      ]
    },
    {
      "id": "002", 
      "title": "Implement user authentication",
      "description": "Setup NextAuth.js with email/password and social login",
      "category": "authentication",
      "priority": "high",
      "estimatedTime": "6 hours",
      "dependencies": ["001"],
      "subtasks": [
        "Install NextAuth.js",
        "Configure providers",
        "Create login/register pages",
        "Setup session management"
      ]
    }
  ]
}
```

### **Bài 8.5: Task Management Workflow**

**Common Commands:**

**List Tasks:**
```bash
# All tasks
task-master list

# Filter by status  
task-master list --status pending

# Filter by priority
task-master list --priority high
```

**Show Next Task:**
```bash
task-master next
```

**Show Specific Tasks:**
```bash
# Single task
task-master show 1

# Multiple tasks
task-master show 1,3,5
```

**Research Command:**
```bash
task-master research "What are the latest best practices for JWT authentication in Next.js?"
```

**Generate Task Files:**
```bash
# Generate implementation files for tasks
task-master generate

# Generate specific task
task-master generate --task 002
```

### **Bài 8.6: Advanced Task Master Features**

**Multi-Phase Task Breakdown:**

**Planning Phase:**
```bash
task-master phase planning
# Generates:
# - Architecture decisions
# - Technology stack validation  
# - Risk assessment
# - Timeline estimation
```

**Development Phase:**
```bash
task-master phase development
# Generates:
# - Implementation tasks
# - Code templates
# - Test specifications
# - Integration points
```

**Review Phase:**
```bash
task-master phase review  
# Generates:
# - Code review checklist
# - Testing strategies
# - Performance benchmarks
# - Security audit items
```

**Testing Phase:**
```bash
task-master phase testing
# Generates:
# - Unit test templates
# - Integration test scenarios
# - E2E test cases
# - Performance test plans
```

### **Bài 8.7: Integration với Cursor AI**

**Cursor Integration Setup:**
```json
// .cursor-rules
Task Master Integration:

WHEN starting new task:
1. Run: task-master show <task-id>
2. Review task requirements and subtasks
3. Generate implementation following task specs
4. Update task progress automatically

TASK TEMPLATES:
- Use task description as implementation guide
- Follow estimated time constraints
- Check dependencies before starting
- Generate tests according to task requirements

AI WORKFLOW:
- Reference task context in prompts
- Break complex tasks into subtasks
- Maintain consistency across related tasks
- Auto-update task status on completion
```

**Workflow Example:**
```bash
# Get next task
task-master next
# Output: Task 003: Implement product catalog API

# Work with Cursor
# Cursor prompt: "Implement task 003 - product catalog API with CRUD operations, following the task requirements and using hexagonal architecture"

# Update progress  
task-master update 003 --status in-progress
task-master update 003 --status completed
```

### **Bài 8.8: Thực hành tổng hợp**

**Scenario**: Xây dựng Blog Platform

**Bước 1**: Initialize project
```bash
task-master init --rules cursor,windsurf,typescript,next.js
```

**Bước 2**: Create PRD
```markdown
# Blog Platform Requirements

## Core Features
1. User Management (registration, profiles)
2. Post Management (create, edit, publish)  
3. Comment System
4. Tag/Category System
5. Search Functionality

## Tech Stack
- Frontend: Next.js 14 with TypeScript
- Backend: Next.js API routes
- Database: PostgreSQL with Prisma
- Authentication: NextAuth.js
- Styling: Tailwind CSS
```

**Bước 3**: Parse và generate tasks
```bash
task-master parse-prd blog-requirements.md
task-master list
```

**Bước 4**: Work through tasks với Cursor
```bash
# Start with setup tasks
task-master next
# Use Cursor với Task Master context
# Complete tasks theo workflow đã định
```

**Kết quả**: 
- Automated task breakdown từ PRD
- Structured development workflow  
- Consistent implementation patterns
- Integrated AI assistance throughout development process

Hệ thống Task Master giúp developers:
- **Reduce planning overhead** bằng automated task generation
- **Maintain focus** với clear task priorities  
- **Ensure consistency** qua standardized workflows
- **Accelerate development** với AI-assisted implementation

Sources
[1] Mô hình Ngôn ngữ lớn (LLM) là gì? - VNG Cloud https://vngcloud.vn/vi/blog/what-are-large-language-models
[2] LLM là gì? Tìm hiểu mô hình Large Language Model trong AI hiện đại https://base.vn/blog/llm-la-gi/
[3] Large Language Model là gì ? Giải thích dễ hiểu | 200Lab Blog https://200lab.io/blog/large-language-model-la-gi
[4] Tokenizers - Hugging Face LLM Course https://huggingface.co/learn/llm-course/vi/chapter2/4
[5] AI Agent Roadmap: Ảnh Hưởng Của Tokenization Đến Hiệu Năng ... https://tuyendung.evotek.vn/ai-agent-roadmap-anh-huong-cua-tokenization-den-hieu-nang-cua-ai-agent/
[6] Hướng dẫn Prompt Engineering dành cho Developer | 200Lab Blog https://200lab.io/blog/huong-dan-prompt-engineering-danh-cho-developer/
[7] Các Kỹ Thuật Tạo Prompt Đỉnh Cao Giúp Làm Chủ AI - Uplift https://uplift.vn/cac-ky-thuat-tao-prompt-dinh-cao-giup-lam-chu-ai-huong-dan-chi-tiet-vi-du-thuc-te
[8] LM Studio: Công cụ chạy LLM Local Nhanh chóng Dễ dàng cho ... https://200lab.io/blog/lm-studio-la-gi/
[9] Cursor vs Windsurf: So sánh và phối hợp cả hai cho việc phát triển ... https://200lab.io/blog/cursor-vs-windsurf-so-sanh-va-phoi-hop
[10] Cursor vs Windsurf vs Cline: Which AI Dev Tool Is Right for You? https://uibakery.io/blog/cursor-vs-windsurf-vs-cline
[11] Hexagonal Architecture là gì và ứng dụng của nó - AI Design https://aithietke.com/hexagonal-architecture-la-gi-va-ung-dung-cua-no/
[12] Model Context Protocol - Wikipedia https://en.wikipedia.org/wiki/Model_Context_Protocol
[13] Introducing the Model Context Protocol - Anthropic https://www.anthropic.com/news/model-context-protocol
[14] Model Context Protocol (MCP) Tutorial: Build Your First MCP Server ... https://towardsdatascience.com/model-context-protocol-mcp-tutorial-build-your-first-mcp-server-in-6-steps/
[15] bernaferrari/FigmaToCode: Generate responsive pages ... - GitHub https://github.com/bernaferrari/FigmaToCode
[16] FreePeak/db-mcp-server: A powerful multi-database server ... - GitHub https://github.com/FreePeak/db-mcp-server
[17] Can I connect Model Context Protocol (MCP) servers to databases ... https://milvus.io/ai-quick-reference/can-i-connect-model-context-protocol-mcp-servers-to-databases-or-file-systems
[18] Task Master: AI-Powered Task Management for Developers https://mcpmarket.com/server/task-master
[19] GitHub - eyaltoledano/claude-task-master https://github.com/eyaltoledano/claude-task-master
[20] LM Studio: Cách Dễ Dàng và Tốt Nhất để Chạy LLMs Cục Bộ https://www.toolify.ai/vi/ai-news-vn/lm-studio-cch-d-dng-v-tt-nht-chy-llms-cc-b-3092744
[21] Sức mạnh đặc biệt của tokenizer nhanh - Hugging Face LLM Course https://huggingface.co/learn/llm-course/vi/chapter6/3
[22] 5 Best AI Tools for Developers 2025 - Strapi https://strapi.io/blog/top-ai-tools-for-developers
[23] Prompt Engineering là gì? Lợi ích của việc áp dụng ... - FPT Shop https://fptshop.com.vn/tin-tuc/danh-gia/prompt-engineering-la-gi-185725
[24] AI Code Tools: The Ultimate Guide in 2025 - CodeSubmit https://codesubmit.io/blog/ai-code-tools/
[25] 10 Best AI Tools for Coding in 2025 | Milestone https://mstone.ai/blog/best-ai-tools-coding/
[26] Hexagonal Architecture là gì và ứng dụng của nó - Viblo https://viblo.asia/p/hexagonal-architecture-la-gi-va-ung-dung-cua-no-4dbZNR88ZYM
[27] Kiến trúc Hexagonal trong DDD - VN GEEKS https://vngeeks.com/kien-truc-hexagonal-trong-ddd/
[28] Optimization - Tối ưu hóa hiệu năng trong lập trình Vi điều khiển https://tapit.vn/optimization-toi-uu-hoa-hieu-nang-trong-lap-trinh-vi-dieu-khien/
[29] Model Optimization là gì? Tìm hiểu về tối ưu hóa mô hình, các kỹ ... https://statio.vn/blog/model-optimization-la-gi-tim-hieu-ve-toi-uu-hoa-mo-hinh-cac-ky-thuat-pho-bien-va-loi-ich-cho-hieu-suat-ai
[30] AI-Powered Figma to Code - Builder.io https://www.builder.io/figma-to-code
[31] Optimization là gì? Các nghiên cứu khoa học về Optimization https://scholarhub.vn/topic/optimization
[32] Builder.io - AI-Powered Figma to Code (React, Vue, Tailwind, & more) https://www.figma.com/community/plugin/747985167520967365/builder-io-ai-powered-figma-to-code-react-vue-tailwind-more
[33] LM Studio Là Gì? Hướng Dẫn Cài Đặt và Sử Dụng Mô Hình Ngôn ... https://thuegpu.vn/lm-studio-la-gi-huong-dan-cai-dat-va-su-dung-llm/
[34] Large language model là gì? Tất cả những điều bạn cần biết về mô ... https://vinbigdata.com/kham-pha/large-language-model-la-gi-tat-ca-nhung-dieu-ban-can-biet-ve-mo-hinh-ngon-ngu-nay.html
[35] IDE AI nào tốt nhất? Windsurf vs Cursor vs Cline vs GitHub Copilot https://apidog.com/vi/blog/windsurf-cursor-cline-github-copilot-vi/
[36] Cursor, Windsurf hay Cline – Cái nào ngon nhất hiện giờ nhỉ? - Reddit https://www.reddit.com/r/ChatGPTCoding/comments/1hhh1tc/cursor_vs_windsurf_vs_cline_whats_the_best_at_the/?tl=vi
[37] Build ANYTHING With TaskMaster + Context7 + Claude ... - YouTube https://www.youtube.com/watch?v=Ox-8X9vowEc
[38] Ports & Adapters Architecture - Craftsmanship - WordPress.com https://edwardthienhoang.wordpress.com/2018/01/18/ports-adapters-architecture/
[39] Unified Model Context Protocol (MCP) Server for Databases https://mindsdb.com/unified-model-context-protocol-mcp-server-for-databases
[40] Tối ưu hoá là gì? Các nghiên cứu khoa học về Tối ưu hoá https://scholarhub.vn/topic/t%E1%BB%91i%20%C6%B0u%20ho%C3%A1
