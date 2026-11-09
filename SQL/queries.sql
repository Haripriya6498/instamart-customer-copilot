
-- QUERY 1: Find completed orders
SELECT *
FROM orders
WHERE status = 'completed';


-- QUERY 2: Join orders with customers and products
SELECT
    o.order_id,
    c.customer_id,
    c.city,
    p.product_name,
    o.quantity,
    o.order_amount
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id;


-- QUERY 3: Category-wise revenue
SELECT
    p.category,
    SUM(o.order_amount) AS total_revenue
FROM orders o
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
HAVING SUM(o.order_amount) > 0;


-- QUERY 4: Orders above average order amount
SELECT order_id, order_amount
FROM orders
WHERE order_amount > (
    SELECT AVG(order_amount)
    FROM orders
);


-- QUERY 5: Customers who have not placed orders
SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;


-- QUERY 6: Monthly revenue using CTE
WITH monthly_sales AS (
    SELECT
        strftime('%Y-%m', order_ts) AS month,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY strftime('%Y-%m', order_ts)
)
SELECT *
FROM monthly_sales
ORDER BY month;


-- QUERY 7: Running revenue using window function
SELECT
    order_id,
    order_ts,
    order_amount,
    SUM(order_amount) OVER (
        ORDER BY order_ts, order_id
        ROWS BETWEEN UNBOUNDED PRECEDING
        AND CURRENT ROW
    ) AS running_revenue
FROM orders
WHERE status = 'completed';
