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

## 滑动窗口
滑动窗口模式是用于在给定数组或链表的特定窗口大小上执行所需的操作，比如寻找包含所有 1 的最长子数组。从第一个元素开始滑动窗口并逐个元素地向右滑，并根据你所求解的问题调整窗口的长度。在某些情况下窗口大小会保持恒定，在其它情况下窗口大小会增大或减小。

下面是一些你可以用来确定给定问题可能需要滑动窗口的方法：
* 问题的输入是一种线性数据结构，比如链表、数组或字符串
* 你被要求查找最长/最短的子字符串、子数组或所需的值
  
你可以使用滑动窗口模式处理的常见问题：
* 大小为 K 的子数组的最大和（简单）
* 带有 K 个不同字符的最长子字符串（中等）  ----*采用滑动窗口，用HashMap记录窗口中间的字符串是否满足要求*
* 寻找字符相同但排序不一样的字符串（困难）

<img src="https://user-images.githubusercontent.com/28688510/155357541-3de82267-d6f7-4ffb-a69d-24ca39166cf4.png" width="600">

这题与带有 K 个不同字符的最长子字符串类似，采用滑动窗口与hashset记录双指针中间的字符串是否有相同的字符，在每一步的操作中，我们会将左指针向右移动一格，表示**我们开始枚举下一个字符作为起始位置**，然后我们可以不断地向右移动右指针，但需要保证这两个指针对应的子串中没有重复的字符。在移动结束后，这个子串就对应着 **以左指针开始的，不包含重复字符的最长子串**。我们记录下这个子串的长度，最后找到最大值即可。


```c++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        // 哈希集合，记录每个字符是否出现过
        unordered_set<char> occ;
        int n = s.size();
        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        int rk = -1, ans = 0;
        // 枚举左指针的位置，初始值隐性地表示为 -1
        for (int i = 0; i < n; ++i) {
            if (i != 0) {
                // 左指针向右移动一格，移除一个字符
                occ.erase(s[i - 1]);
            }
            while (rk + 1 < n && !occ.count(s[rk + 1])) {
                // 不断地移动右指针
                occ.insert(s[rk + 1]);
                ++rk;
            }
            // 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1);
        }
        return ans;
    }
};
```

<img src="https://user-images.githubusercontent.com/28688510/155359599-5ca15f08-823b-4afb-9d53-39bbc540cbb1.png" width="600">
<img src="https://user-images.githubusercontent.com/28688510/155360290-8e5a5be7-2b14-40e9-8686-76ab381b7c14.gif" width="600">

```c++
class Solution {
public:
    string minWindow(string s, string t) {
        if (s.size() < t.size()) return "";
        int map[128];
        // 遍历字符串 t，初始化每个字母的次数
        for (int i = 0; i < t.size(); i++) {
            map[t[i]]++;
        }
        int left = 0; // 左指针
        int right = 0; // 右指针
        int ans_left = 0; // 保存最小窗口的左边界
        int ans_right = -1; // 保存最小窗口的右边界
        int ans_len = INT_MAX; // 当前最小窗口的长度
        int count = t.size();
        // 遍历字符串 s
        while (right < s.size()) {
            // 当前的字母次数减一
            map[s[right]]--;

            //代表当前符合了一个字母
            if (map[s[right]] >= 0) {
                count--;
            }
            // 开始移动左指针，减小窗口
            while (count == 0) { // 如果当前窗口包含所有字母，就进入循环
                // 当前窗口大小
                int temp_len = right - left + 1;
                // 如果当前窗口更小，则更新相应变量
                if (temp_len < ans_len) {
                    ans_left = left;
                    ans_right = right;
                    ans_len = temp_len;
                }
                // 因为要把当前字母移除，所有相应次数要加 1
                map[s[left]]++;
                //此时的 map[key] 大于 0 了，表示缺少当前字母了，count++
                if (map[s[left]] > 0) {
                    count++;
                }
                left++; // 左指针右移
            }
            // 右指针右移扩大窗口
            right++;
        }
        return s.substr(ans_left, ans_len);
    }
};
```

<img src="https://user-images.githubusercontent.com/28688510/155360915-793c2d2d-e8a2-4de6-b6d9-d293f341d538.png" width="600">

```c++
class Solution {
public:
    int minSubArrayLen(int s, vector<int>& nums) {
        int len = nums.size();
        if (len == 0) {return 0;}
        int start = 0, end = 0;
        int ans = INT_MAX;
        int sum = 0;
        while (end < len) {
            sum += nums[end];
            while (sum >= s) {
                ans = min(ans, end - start + 1);
                sum -= nums[start];
                start++;
            }
            end++;
        }
        return (ans == INT_MAX)? 0: ans;
    }
};
```

<img src="https://user-images.githubusercontent.com/28688510/155362858-592060e5-fc95-4ec4-ba06-7ac67a2b1430.png" width="600">

这题虽然题目带滑动窗口，但实际与滑动窗口关系不大，这里通过维护一个堆（元素为值和索引）或者双端队列（递减，队首始终为最大值）来解题，当新元素入队列时与队尾比较，如果大于队尾，则队尾元素永远不可能被取为最大值，队尾一直出队直到满足大于入队元素，元素入队后，判断队首元素是否在队列中，如果不在，队首也要出队。这个过程结束后，队首元素即为当前滑动窗口的最大值。

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if (k == 1) return nums;
        // 双端队列，保持队列的递减性，队头元素即为滑动窗口的最大值
        deque<int> q;
        int n = nums.size();

        // 首先将前k个数字挨个入队
        // 入队时如果大于队尾元素则将队尾元素出队，说明永远也不可能取到他
        for (int i = 0; i < k; i++) {
            while (!q.empty() && nums[i] > nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
        }

        vector<int> res = {nums[q.front()]};

        // 其次将k个后的数字挨个入队
        for (int i = k; i < n; i++) {
            // 入队时如果大于队尾元素则将队尾元素出队，说明永远也不可能取到他
            while (!q.empty() && nums[i] > nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
            // 将队头的数字出队直到当前滑动窗口有效范围内
            while (q.front() <= i - k) {
                q.pop_front();
            }
            res.push_back(nums[q.front()]);
        }
        return res;
    }
};
```

<img src="https://user-images.githubusercontent.com/28688510/155363919-504267ee-fc57-445d-981c-f38422ee3378.png" width="600">

此题即对应上面所说的第三种题型，思路一样，维护一个hash表用于保存p的字母个数，滑动窗口解题

```c++
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int n = s.size();
        int m = p.size();

        // 维护一个hash表用于保存p的字母个数
        vector<int> hash(26);
        for (auto c : p) {
            hash[c - 'a']++;
        }

        vector<int> res;
        for (int l = 0, r = 0; r < n; r++) {
            // 每次让r往右走，并减去当前字母一次
            hash[s[r] - 'a']--;
            // 当遇到不是p的字母或者p的字母减的次数多了，则说明[l, r]不是p的异位词
            // 令l右移到r处，并将hash表恢复到正常情况
            while (hash[s[r] - 'a'] < 0) {
                hash[s[l] - 'a']++;
                l++;
            }
            // 如果哈希表正常（均大于等于0）并[l, r]长度等于p的长度，则说明[l, r]是p的异位词
            if (r - l + 1 == m) res.push_back(l);
        }
        return res;
    }
};
```


