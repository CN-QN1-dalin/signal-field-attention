-- Agent Reviews API Tests
-- 测试评价系统功能

-- 1. 插入评价
INSERT INTO reviews (agent_id, user_id, rating, comment)
VALUES (
    (SELECT id FROM agents WHERE slug = 'code-assistant'),
    (SELECT id FROM users WHERE username = 'dalinos'),
    5,
    '非常棒的编程助手！'
);

-- 2. 验证评价计数更新
SELECT rating, rating_count FROM agents WHERE slug = 'code-assistant';
-- 预期: rating=5.00, rating_count=1

-- 3. 插入第二条评价
INSERT INTO reviews (agent_id, user_id, rating, comment)
VALUES (
    (SELECT id FROM agents WHERE slug = 'code-assistant'),
    (SELECT id FROM users WHERE username = 'admin'),
    4,
    '很好，但还有改进空间'
);

-- 4. 验证平均分更新
SELECT rating, rating_count FROM agents WHERE slug = 'code-assistant';
-- 预期: rating=4.50, rating_count=2

-- 5. 测试同一用户不能重复评价
INSERT INTO reviews (agent_id, user_id, rating, comment)
VALUES (
    (SELECT id FROM agents WHERE slug = 'code-assistant'),
    (SELECT id FROM users WHERE username = 'dalinos'),
    3,
    '重复评价应该失败'
);
-- 预期: 违反唯一索引约束

-- 6. 测试评价范围 (1-5)
INSERT INTO reviews (agent_id, user_id, rating, comment)
VALUES (
    (SELECT id FROM agents WHERE slug = 'code-assistant'),
    (SELECT id FROM users WHERE username = 'admin'),
    6,
    '超出范围应该失败'
);
-- 预期: 违反 CHECK 约束

-- 7. 查询 Agent 的所有评价
SELECT r.rating, r.comment, u.username, r.created_at
FROM reviews r
JOIN users u ON r.user_id = u.id
WHERE r.agent_id = (SELECT id FROM agents WHERE slug = 'code-assistant')
ORDER BY r.created_at DESC;

-- 8. 测试删除评价后重新计算
DELETE FROM reviews WHERE agent_id = (SELECT id FROM agents WHERE slug = 'code-assistant') AND user_id = (SELECT id FROM users WHERE username = 'admin');
SELECT rating, rating_count FROM agents WHERE slug = 'code-assistant';
-- 预期: rating=5.00, rating_count=1
