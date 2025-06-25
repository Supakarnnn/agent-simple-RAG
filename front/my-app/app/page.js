// pages/index.js
"use client"
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import styles from './page.module.css';

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { role: 'human', content: input };
    setMessages([...messages, userMessage]);
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ messages: [...messages, userMessage] }),
      });
      const data = await response.json();
      setMessages([...messages, userMessage, { role: 'ai', content: data.response }]);
      setInput('');
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.chatBox}>
        <h1 className={styles.title}>Chat with AI</h1>
        <div className={styles.messages}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={msg.role === 'human' ? styles.humanMessage : styles.aiMessage}
            >
              <ReactMarkdown>{msg.content}</ReactMarkdown>
              <div className={styles.tag}>{msg.role === 'human' ? 'User' : 'AI'}</div>
            </div>
          ))}
        </div>
        <div className={styles.inputArea}>
          <input
            type="text"
            className={styles.input}
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className={styles.sendButton}
          >
            {loading ? '...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
