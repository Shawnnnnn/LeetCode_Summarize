# LeetCode_Summarize
这个仓库用来记录自己的力扣刷题以及进行总结，总结类似的题型，避免无效刷题

## 单调栈
“在一维数组中找第一个满足某种条件的数”的场景就是典型的单调栈应用场景  
从名字也可以看出, 它最大的特点就是单调, 也就是**栈中的元素要么递增, 要么递减, 如果有新的元素不满足这个特点, 就不断的将栈顶元素出栈, 直到满足为止**, 这就是它最重要的思想.

<img src="https://user-images.githubusercontent.com/28688510/155183056-dba4f006-dad9-446a-a08b-e5ad4cfaa271.png" width="600">

这一题考察的就是单调栈的运用，求柱状图中的最大矩形，要考虑的是如何找到一根柱子的左右两边第一个小于自己高度的柱子，然后即可得到面积
```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        // 重要，要在最右边加个高度为0的柱子
        heights.push_back(0);
        int n = heights.size();
        // s相当于存左边界
        stack<int> s;
        int res = INT_MIN;

        for (int i = 0; i < n; i++) {
            // 右边沿：正好是i（由于单调栈的性质，第i个柱子就是右边第一个矮于A的柱子）
            // 左边沿：单调栈中紧邻A的柱子。（如果A已经出栈，那么左边沿就是A出栈后的栈顶）
            // 当A出栈后，单调栈为空时，那就是说明，A的左边没有比它矮的。左边沿就可以到0.
            while (!s.empty() && heights[i] <= heights[s.top()]) {
                int h = heights[s.top()];
                s.pop();
                int left = s.empty()? -1: s.top();
                res = max(res, h * (i - left - 1));
            }
            s.push(i);
        }

        return res;
    }
};
```

<img src="https://user-images.githubusercontent.com/28688510/155183675-bea1430c-f636-4378-8701-249f558f6a05.png" width="600">

同样，这一题可以看做是多个柱状图中找最大面积，做法与上一题基本一致，区别在于要逐步从第一行到最后一行将矩阵看做柱状图，再重复调用上一题的解题方法即可
```c++
class Solution {
private:
    int res;
public:
    int maximalRectangle(vector<vector<char>>& matrix) {
        int n = matrix.size();
        int m = matrix[0].size();
        for (int i = 0; i < n; i++) {
            vector<int> tmp(m);
            for (int k = 0; k < m; k++) {
                int j = i;
                while (j < n && matrix[j][k] == '1') {
                    j++;
                }
                tmp[k] = j - i;
            }

            solve(tmp);
        }
        return res;
    }

    void solve(vector<int> heights) {
        heights.push_back(0);
        int n = heights.size();
        stack<int> s;

        for (int i = 0; i < n; i++) {
            while (!s.empty() && heights[i] < heights[s.top()]) {
                int h = heights[s.top()];
                s.pop();
                int left = s.empty()? -1: s.top();

                res = max(res, h * (i - left - 1));
            }
            s.push(i);
        }
    }
};
```

<img src="https://user-images.githubusercontent.com/28688510/155186443-7e951105-7714-4bac-b8b7-1ab266e3f879.png" width="600">
对于下标 i，下雨后水能到达的最大高度等于下标 i 两边的最大高度的最小值，下标 i 处能接的雨水量等于下标 i 处的水能到达的最大高度减去 height[i]。
　

朴素的做法是对于数组 height 中的每个元素，分别向左和向右扫描并记录左边和右边的最大高度，然后计算每个下标位置能接的雨水量。假设数组 height 的长度为 n，该做法需要对每个下标位置使用 O(n) 的时间向两边扫描并得到最大高度，因此总时间复杂度是 O(n^2)，但是**向左和向右扫描并记录左边和右边的最大高度**这一点很明显满足单调栈的特性，因此我们可以用单调栈来做。
  
维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 height 中的元素递减。遍历柱子，当不满足当前遍历到的柱子小于栈顶柱子的高度时，说明栈顶右边第一个大于他的柱子出现了，然后栈顶出栈，此时的栈顶就是左边第一个大于出栈柱子的柱子，因此可以计算出栈柱子的接水面积了。

<img src="https://user-images.githubusercontent.com/28688510/155190481-f783337b-da43-4709-aacf-de62b04c574d.png" width="600">
<img src="https://user-images.githubusercontent.com/28688510/155190603-e5ecd101-058c-4142-9f1d-edd44ae11c60.png" width="600">

```c++
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0;
        stack<int> stk;
        int n = height.size();
        for (int i = 0; i < n; ++i) {
            while (!stk.empty() && height[i] > height[stk.top()]) {
                int top = stk.top();
                stk.pop();
                if (stk.empty()) {
                    break;
                }
                int left = stk.top();
                int currWidth = i - left - 1;
                int currHeight = min(height[left], height[i]) - height[top];
                ans += currWidth * currHeight;
            }
            stk.push(i);
        }
        return ans;
    }
};
```
  
此题还可以采用**双指针**来做，维护两个指针 left 和 right，以及两个变量 leftMax 和 rightMax，初始时 left=0,right=n-1,leftMax=0,rightMax=0。指针 left 只会向右移动，指针 right 只会向左移动，在移动指针的过程中维护两个变量 leftMax 和 rightMax 的值。
<img src="https://user-images.githubusercontent.com/28688510/155192585-5f6fc5b2-4f81-40f5-9db0-872b9fa0b847.png" width="600">

```c++
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0;
        int left = 0, right = height.size() - 1;
        int leftMax = 0, rightMax = 0;
        while (left < right) {
            leftMax = max(leftMax, height[left]);
            rightMax = max(rightMax, height[right]);
            if (height[left] < height[right]) {
                ans += leftMax - height[left];
                ++left;
            } else {
                ans += rightMax - height[right];
                --right;
            }
        }
        return ans;
    }
};
```

