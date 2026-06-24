// ==================== API Client Library ====================
// 作者: 元宝 (前端专家)
// 日期: 2026-06-24
// 功能: 统一 API 调用封装

const API_BASE = 'http://localhost:3000/api/v1';

class DalinAPI {
    constructor() {
        this.token = localStorage.getItem('access_token') || null;
        this.refreshToken = localStorage.getItem('refresh_token') || null;
        this.user = JSON.parse(localStorage.getItem('user') || 'null');
    }

    // ==================== Auth ====================
    
    async register(username, email, password, passwordConfirm) {
        const res = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password, password_confirm: passwordConfirm })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '注册失败');
        return data;
    }

    async login(email, password) {
        const res = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '登录失败');
        
        this.token = data.access_token;
        this.refreshToken = data.refresh_token;
        this.user = data.user;
        
        localStorage.setItem('access_token', this.token);
        localStorage.setItem('refresh_token', this.refreshToken);
        localStorage.setItem('user', JSON.stringify(this.user));
        
        return data;
    }

    async logout() {
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
        this.token = null;
        this.refreshToken = null;
        this.user = null;
    }

    isAuthenticated() {
        return !!this.token;
    }

    // ==================== Agents ====================
    
    async listAgents({ category, tag, search, sort, page = 1, limit = 20 } = {}) {
        const params = new URLSearchParams();
        if (category) params.set('category', category);
        if (tag) params.set('tag', tag);
        if (search) params.set('search', search);
        if (sort) params.set('sort', sort);
        params.set('page', page);
        params.set('limit', limit);
        
        const res = await fetch(`${API_BASE}/agents?${params}`);
        return res.json();
    }

    async getAgent(slug) {
        const res = await fetch(`${API_BASE}/agents/${slug}`);
        return res.json();
    }

    async createAgent(data) {
        const res = await fetch(`${API_BASE}/agents`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify(data)
        });
        return res.json();
    }

    async deleteAgent(slug) {
        const res = await fetch(`${API_BASE}/agents/${slug}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${this.token}` }
        });
        return res.json();
    }

    // ==================== Reviews ====================
    
    async createReview(agentSlug, rating, comment) {
        const res = await fetch(`${API_BASE}/agents/${agentSlug}/reviews`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({ rating, comment })
        });
        return res.json();
    }

    async getReviews(agentSlug) {
        const res = await fetch(`${API_BASE}/agents/${agentSlug}/reviews`);
        return res.json();
    }

    // ==================== Compile ====================
    
    async compile(code, version) {
        const res = await fetch(`${API_BASE}/compile`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({ code, version })
        });
        return res.json();
    }

    // ==================== Dashboard ====================
    
    async getAgentStatuses() {
        const res = await fetch(`${API_BASE}/dashboard/agents`);
        return res.json();
    }

    async getWalletSummary(userId) {
        const res = await fetch(`${API_BASE}/dashboard/wallet/${userId}`);
        return res.json();
    }

    async getTransactions(userId) {
        const res = await fetch(`${API_BASE}/dashboard/transactions/${userId}`);
        return res.json();
    }

    // ==================== Social (灵光一现) ====================
    
    async createProfile(profileData) {
        const res = await fetch(`${API_BASE}/social/profile`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify(profileData)
        });
        return res.json();
    }

    async getProfiles({ page = 1, limit = 20, interests = [] } = {}) {
        const params = new URLSearchParams({ page, limit });
        interests.forEach(i => params.append('interests', i));
        
        const res = await fetch(`${API_BASE}/social/profiles?${params}`);
        return res.json();
    }

    async likeProfile(profileId) {
        const res = await fetch(`${API_BASE}/social/likes`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({ profile_id: profileId })
        });
        return res.json();
    }

    async getMatches() {
        const res = await fetch(`${API_BASE}/social/matches`, {
            headers: { 'Authorization': `Bearer ${this.token}` }
        });
        return res.json();
    }

    async sendMessage(toUserId, content) {
        const res = await fetch(`${API_BASE}/social/messages`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({ to_user_id: toUserId, content })
        });
        return res.json();
    }

    async getMessages(userId) {
        const res = await fetch(`${API_BASE}/social/messages?user_id=${userId}`, {
            headers: { 'Authorization': `Bearer ${this.token}` }
        });
        return res.json();
    }

    async updateProfile(profileData) {
        const res = await fetch(`${API_BASE}/social/profile`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify(profileData)
        });
        return res.json();
    }

    async flashMessage(content) {
        const res = await fetch(`${API_BASE}/social/flash`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({ content })
        });
        return res.json();
    }

    async getFlashFeed({ limit = 20, page = 1 } = {}) {
        const res = await fetch(`${API_BASE}/social/flash?limit=${limit}&page=${page}`);
        return res.json();
    }
}

// Export singleton
export const api = new DalinAPI();
