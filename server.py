#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""知识蒸馏 APP - 服务端

Serves standalone HTML + provides LLM-powered chat API (RAG + DeepSeek).
"""

import http.server
import socketserver
import json
import re
import socket
import sys
import os
import random

PORT = int(os.environ.get('PORT', '5678'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load knowledge base
with open(os.path.join(BASE_DIR, 'knowledge.json'), 'r', encoding='utf-8') as f:
    kb = json.load(f)

# Read standalone HTML
with open(os.path.join(BASE_DIR, 'standalone.html'), 'r', encoding='utf-8') as f:
    html_content = f.read()

# DeepSeek API key - try env var first, then .env file
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
if not DEEPSEEK_API_KEY:
    env_path = os.path.join(BASE_DIR, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('DEEPSEEK_API_KEY='):
                    DEEPSEEK_API_KEY = line.split('=', 1)[1].strip().strip('"').strip("'")
                    break


# ============================================================
# Knowledge Search (same as before)
# ============================================================

def tokenize(text):
    """Tokenize Chinese text into search tokens."""
    tokens = set()
    for m in re.finditer(r'[一-鿿]{2,5}', text):
        s = m.group()
        for i in range(len(s)):
            for j in range(i + 2, min(i + 4, len(s) + 1)):
                tokens.add(s[i:j])
    for m in re.finditer(r'[a-zA-Z]{2,}', text):
        tokens.add(m.group().lower())
    return tokens


def search_knowledge(query, person='all', top_k=5):
    """Search knowledge base for relevant entries."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    persons = [person] if person != 'all' else ['谢胜子', '曲曲']
    scored = []

    for p in persons:
        entries = kb.get(p, [])
        for entry in entries:
            entry_tokens = set(entry.get('keywords', []))
            overlap = query_tokens & entry_tokens
            if overlap:
                score = len(overlap) * (1 + 0.5 * len(overlap))
                heading_tokens = tokenize(entry.get('heading', ''))
                heading_overlap = query_tokens & heading_tokens
                score += len(heading_overlap) * 3
                scored.append((score, entry, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ============================================================
# LLM Integration (DeepSeek)
# ============================================================

def build_persona_context(results, person):
    """Build a knowledge context string from search results for LLM prompt."""
    if not results:
        return ""

    parts = []
    seen_headings = set()
    for score, entry, p in results:
        heading = entry.get('heading', '')
        domain = entry.get('domain', '')
        text = entry.get('text', '')
        if heading in seen_headings or not text:
            continue
        seen_headings.add(heading)
        lines = [l.strip() for l in text.split('\n')
                 if l.strip() and not l.startswith('---') and not l.startswith('[')]
        display = '\n'.join(lines[:20])[:1200]
        parts.append(f"【{domain} - {heading}】\n{display}")

    return '\n\n'.join(parts)


def build_prompt(query, context, person):
    """Build the system + user prompt for the LLM."""

    if person == '谢胜子':
        system_prompt = """你现在是谢胜子本人。你是自媒体创业领域的头部博主，个人IP导师。

这不是角色扮演，你要用你的认知体系来回答问题。

## 你的思维方式（这才是核心）
你不是在"输出观点"，你是在用你的分析框架拆解问题。你的框架是：

1. **看本质**：任何问题先问"这事的本质是什么"，不要停留在表面
2. **分层次**：把问题拆成几个层面——认知层、方法层、执行层
3. **找关联**：各个层面之间是什么关系？基础是什么？进阶是什么？顶层是什么？
4. **给方法**：不只是说"是什么"，要说"怎么做"。给具体的方法论框架

## 你的认知体系（回答的根基）
- IP的本质是认知差+表达力+信任感，不是网红
- 做IP分三层：知名度→信任度→溢价能力，顺序不能乱
- 赛道定位的核心是"找到自己的独特属性组合"，不是追热点
- 内容创作要建立你的"三个体系"：阅读体系、写作体系、认知体系
- 变现的四个圈层：卖产品→卖服务→卖圈子→卖身份
- 个人IP vs 自媒体号：做号追流量，做IP追影响力

## 你的说话方式（自然就好）
- 口语化，像朋友聊天时输出思考，不是念稿
- 可以用"好""我告诉你""你懂我意思吧""说白了"
- 可以反问："你发现没有？""你想想是不是这个道理？"
- 不要用任何markdown格式（不要加粗、列表序号、标题）
- 不要说"首先其次最后""第一第二第三"
- 不要用"综上所述"这类书面语

## 给你的参考内容（你积累的知识和经验）
以下是你多年积累的认知体系，回答时融会贯通地使用它们：

{context}

## 回答要求
- 真正用你的认知框架来分析问题，给出有深度的见解
- 如果是具体问题，给方法论+执行建议
- 如果参考内容里有相关框架，自然地运用它们，不要生硬引用
- 长度不限，说到透彻为止"""

    elif person == '曲曲':
        system_prompt = """你现在是曲曲大女人本人。你是两性关系领域的顶流IP，人间清醒。

你不是来哄人的，你是来把人骂醒的。

## 你的思维方式（这才是核心）
你看任何问题的角度都和普通人不一样：

1. 关系的本质是交换——先算清楚"你图什么、他图什么"，别跟我扯感情
2. 价值决定一切——你的价值决定你的待遇，不是你的付出决定你的待遇。你把自己放得多低，别人就把你踩得多低
3. 筛选大于改变——成年人只筛选不教育，不要指望改变任何人。你觉得不舒服？那是你的问题，不是他的问题
4. 强者思维 vs 弱者思维——弱者问"他为什么这样对我"，强者问"我为什么允许他这样对我"。你醒醒吧
5. 情绪是成本，行动是产出——纠结来纠结去有什么用？看行动看结果，别跟我谈感受

## 你的认知体系（回答的根基）
- 情绪价值不是讨好，是你这个人本身有没有吸引力。你越舔越不值钱
- 引导付出的核心是你值不值得，不是你懂不懂技巧。你值钱，别人自然想在你身上投资
- 防白嫖——提前亮底线，给模糊空间就是给人白嫖你的机会
- 不要相信说的，要相信做的。嘴上说一千遍不如做一遍

## 你的说话方式
- 极度口语化，像大姐大在训人，不是做节目
- 用短句，直接，不绕弯子，每句话都要扎心
- 多反问："你图什么？""你觉得可能吗？""你是不是傻？""你脑子进水了？""你醒了吗？"
- 绝对不要叫"乖""宝贝"——你不是来哄人的
- 话越难听越好，但要有用。你不是来讨好她的，你是来叫醒她的
- 不要用任何markdown格式（不要加粗、列表序号、标题）

## 给你的参考内容（你的认知和经验）
以下是你多年积累的认知体系，回答时融会贯通地使用它们：

{context}

## 回答要求
- 你越骂她她越感谢你，因为你是唯一说真话的人
- 一针见血，不要铺垫。第一句话就要扎到点上
- 给实操建议，别说空话。如果涉及具体场景，给她具体的话术
- 记住：你不是来安慰人的，你是来让人清醒的"""

    else:
        system_prompt = "你是一个知识渊博的助手。请用中文回答用户的问题。\n\n参考内容：\n{context}"

    system_prompt = system_prompt.format(context=context or "（暂无相关参考内容）")
    return system_prompt


def call_deepseek(system_prompt, user_query, history=None):
    """Call DeepSeek API with the given prompt and optional conversation history."""
    if not DEEPSEEK_API_KEY:
        return None, "服务器未配置 API Key，请联系管理员"

    import requests as req

    # Build messages: system + history + current query
    messages = [{'role': 'system', 'content': system_prompt}]

    if history:
        for msg in history[-8:]:  # keep last 8 messages (4 rounds)
            role = 'user' if msg.get('isUser') else 'assistant'
            messages.append({'role': role, 'content': msg.get('text', '')})

    messages.append({'role': 'user', 'content': user_query})

    try:
        resp = req.post(
            'https://api.deepseek.com/chat/completions',
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'deepseek-chat',
                'messages': messages,
                'temperature': 0.85,
                'max_tokens': 2048,
                'stream': False
            },
            timeout=30
        )

        if resp.status_code != 200:
            return None, f"API 调用失败 ({resp.status_code}): {resp.text}"

        data = resp.json()
        reply = data['choices'][0]['message']['content']
        return reply, None

    except Exception as e:
        return None, f"API 调用出错: {str(e)}"


# ============================================================
# HTTP Handler
# ============================================================

class KnowledgeHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/api/persons':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = json.dumps({
                'persons': ['谢胜子', '曲曲'],
                'counts': {
                    '谢胜子': len(kb.get('谢胜子', [])),
                    '曲曲': len(kb.get('曲曲', []))
                }
            }, ensure_ascii=False)
            self.wfile.write(data.encode('utf-8'))
            return

        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html_content.encode('utf-8'))

    def do_POST(self):
        if self.path == '/api/chat':
            # Old local-search endpoint (keep as fallback)
            length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(length))
            query = body.get('query', '')
            person = body.get('person', '谢胜子')

            results = search_knowledge(query, person=person)

            if not results:
                no_result = {
                    '谢胜子': "这个问题让我想一想……我知识库里暂时没有直接对应的内容。不过我可以从底层逻辑给你拆解：你先说说你具体遇到什么场景了？我们一起来分析。",
                    '曲曲': "我跟你说，你这个问题吧，它不是没有答案，是你问的方式不对。你先告诉我你具体什么情况，越具体越好，我才能给你拆解。别给我整虚的，直接说事。"
                }
                reply = no_result.get(person, "没找到相关内容。")
                refs = []
            elif person == '谢胜子':
                reply, refs = self._format_xie_shengzi(results)
            else:
                reply, refs = self._format_ququ(results)

            self._send_json({'reply': reply, 'references': refs})
            return

        elif self.path == '/api/chat-llm':
            # LLM-powered endpoint
            length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(length))
            query = body.get('query', '')
            person = body.get('person', '谢胜子')
            history = body.get('history', [])  # conversation history

            if not DEEPSEEK_API_KEY:
                # Fallback to local search if no API key
                results = search_knowledge(query, person=person)
                if person == '谢胜子':
                    reply, refs = self._format_xie_shengzi(results)
                else:
                    reply, refs = self._format_ququ(results)
                self._send_json({
                    'reply': reply,
                    'references': refs,
                    'mode': 'local'
                })
                return

            # RAG: search knowledge base for context
            results = search_knowledge(query, person=person, top_k=10)
            context = build_persona_context(results, person)
            system_prompt = build_prompt(query, context, person)

            # Call DeepSeek
            reply, error = call_deepseek(system_prompt, query, history=history)

            if error:
                # Fallback to local formatting on error
                if person == '谢胜子':
                    reply, refs = self._format_xie_shengzi(results)
                else:
                    reply, refs = self._format_ququ(results)
                self._send_json({
                    'reply': reply,
                    'references': refs,
                    'mode': 'local',
                    'error': error
                })
                return

            # Build references
            refs = []
            seen = set()
            for _, entry, _ in results:
                d = entry.get('domain', '')
                if d not in seen:
                    seen.add(d)
                    refs.append({'domain': d, 'heading': entry.get('heading', '')})

            self._send_json({
                'reply': reply,
                'references': refs,
                'mode': 'llm'
            })
            return

        self.send_response(404)
        self.end_headers()

    def _send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def _format_xie_shengzi(self, results):
        """Format search results in 谢胜子's voice (local fallback)."""
        if not results:
            return ("这个问题问得很好，但我知识库里暂时没有直接对应的内容。"
                    "不过没关系，我可以从底层逻辑给你拆解——"
                    "你先说说你具体遇到什么场景了？我来帮你分析。", [])

        references = []
        seen_domains = set()
        for _, entry, _ in results:
            domain = entry.get('domain', '')
            if domain not in seen_domains:
                seen_domains.add(domain)
                references.append({'domain': domain, 'heading': entry.get('heading', '')})

        parts = []
        openings = [
            '好，这个问题值得认真拆解。我给你从几个维度来分析。\n',
            '这个问题其实涉及到几个核心层面，我来给你梳理一下。\n',
            '好。这个话题我可以展开讲讲，你注意听几个关键点。\n'
        ]
        parts.append(random.choice(openings))

        for i, (score, entry, person) in enumerate(results):
            if i >= 4:
                break
            text = entry.get('text', '')
            heading = entry.get('heading', '')
            if heading in ('总览',) or text is None or len(text) < 30:
                continue

            lines = [l for l in text.split('\n') if l.strip() and not l.startswith('---')]
            display = '\n'.join(lines[:15])[:1000]

            if heading and heading != '总览':
                transitions = [
                    f'我们先来看{heading}：\n',
                    f'这里有一个关键概念叫"{heading}"：\n',
                    f'再说{heading}，这个点很重要：\n',
                    f'接下来，{heading}：\n'
                ]
                parts.append(random.choice(transitions))

            parts.append(display + '\n\n')

        synths = [
            '所以我给你总结一下：这几个点其实是互相关联的。第一个点是基础，第二个点是进阶，第三个点是顶层。你要做的是先把基础打牢，再往上走。\n\n',
            '你发现没有？这几个维度其实是层层递进的。你不能跳过第一步直接做第三步。做个人IP这件事，顺序很重要。\n\n',
            '所以你看，这背后有一个清晰的逻辑链条。我建议你把这个框架保存下来，做内容的时候对照着看。\n\n'
        ]
        parts.append(random.choice(synths))

        closings = [
            '你懂我意思吧？以上是方法论层面的拆解。具体到你自己的情况，还需要结合你的实际赛道和资源来落地。如果你有更具体的问题，可以继续说，我帮你进一步分析。',
            '说白了，这些都是经过验证的方法论。但理论是理论，真正做的时候一定要结合你自己的实际情况。每个人的赛道、属性、资源都不一样。你先把这几个点消化一下，有具体问题我们再深挖。',
            '好，今天就先拆到这里。你先把这几个关键点理解透，然后去实践。实践中遇到什么问题，随时来问我。'
        ]
        parts.append(random.choice(closings))

        return ''.join(parts), references

    def _format_ququ(self, results):
        """Format search results in 曲曲's voice (local fallback)."""
        if not results:
            return ("我跟你说，你这个问题吧，不是没有答案，是你问的方式不对。"
                    "你先告诉我你具体什么情况，越具体越好，我才能给你拆解。"
                    "别给我整虚的，直接说事。", [])

        best = results[0]
        entry = best[1]
        text = entry.get('text', '')
        lines = [l for l in text.split('\n') if l.strip()]
        substantive = [l for l in lines if len(l) > 8
                       and not l.startswith('-') and not l.startswith('*') and not l.startswith('#')]

        parts = []
        openings = [
            '好，我告诉你这个事的本质是什么。\n\n',
            '来，我跟你说明白了——\n\n',
            '你听好了，我只说一遍。\n\n'
        ]
        parts.append(random.choice(openings))

        if substantive and len(substantive) >= 3:
            parts.append('\n'.join(substantive[:8]) + '\n\n')
        else:
            parts.append('\n'.join(lines[:8]) + '\n\n')

        punches = [
            '你是不是也觉得这事儿没那么复杂？说白了，就是你想太多了。\n\n',
            '你说你这是何必呢？非要把简单的事情搞复杂。\n\n',
            '你醒了吗？这个逻辑你要是没想通，后面说什么都是白搭。\n\n'
        ]
        parts.append(random.choice(punches))

        if len(results) > 1:
            text2 = results[1][1].get('text', '')
            s2 = [l for l in text2.split('\n')
                  if l.strip() and not l.startswith('-') and not l.startswith('*') and len(l) > 10]
            if s2:
                parts.append('我再给你补一句——\n')
                parts.append('\n'.join(s2[:4]) + '\n\n')

        closings = [
            '你记住，这事儿没有对错，只有强弱。你自己想想是不是这个理儿。',
            '听我的，别在这个问题上纠结了。你把精力放在提升自己上，比什么都强。',
            '我话讲完了。能听进去多少，看你自己的悟性。记住：行动推动决策，别光想不做。',
            '乖，听我的。这个世界上最不值钱的就是情绪，最值钱的是你的行动力。'
        ]
        parts.append(random.choice(closings))

        reply = ''.join(parts)
        references = [{'domain': entry.get('domain', ''), 'heading': entry.get('heading', '')}]
        if len(results) > 1:
            references.append({'domain': results[1][1].get('domain', ''),
                              'heading': results[1][1].get('heading', '')})

        return reply, references

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


# ============================================================
# Threaded server for concurrent requests
# ============================================================

class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Handle requests in separate threads so LLM calls don't block."""
    allow_reuse_address = True
    daemon_threads = True


def get_local_ip():
    """Get local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '192.168.x.x'


if __name__ == '__main__':
    local_ip = get_local_ip()
    has_key = bool(DEEPSEEK_API_KEY)

    print('=' * 55)
    print('  知识蒸馏 APP v2 - LLM 增强版')
    print('=' * 55)
    print(f'  本地访问: http://localhost:{PORT}')
    print(f'  手机访问: http://{local_ip}:{PORT}')
    print(f'  知识库: 谢胜子 {len(kb.get("谢胜子",[]))}条 + 曲曲 {len(kb.get("曲曲",[]))}条')
    print(f'  LLM 状态: {"已连接 DeepSeek" if has_key else "未配置 API Key（使用本地模式）"}')
    print(f'  防火墙已放行, 同一WiFi下可访问')
    print('=' * 55)
    print('  按 Ctrl+C 停止服务器')
    print('=' * 55)
    sys.stdout.flush()

    with ThreadingHTTPServer(('0.0.0.0', PORT), KnowledgeHandler) as httpd:
        httpd.serve_forever()
