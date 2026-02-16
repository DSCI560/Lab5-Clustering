SELECT cluster_id, COUNT(*)
FROM posts
GROUP BY cluster_id
ORDER BY cluster_id;
