# LeetCode_Summarize
这个仓库用来记录自己的力扣刷题以及进行总结，总结类似的题型，避免无效刷题

## 单调栈
“在一维数组中找第一个满足某种条件的数”的场景就是典型的单调栈应用场景  
从名字也可以看出, 它最大的特点就是单调, 也就是**栈中的元素要么递增, 要么递减, 如果有新的元素不满足这个特点, 就不断的将栈顶元素出栈, 直到满足为止**, 这就是它最重要的思想.

<img src="https://user-images.githubusercontent.com/28688510/155183056-dba4f006-dad9-446a-a08b-e5ad4cfaa271.png" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155183675-bea1430c-f636-4378-8701-249f558f6a05.png" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155186443-7e951105-7714-4bac-b8b7-1ab266e3f879.png" width="500">
对于下标 i，下雨后水能到达的最大高度等于下标 i 两边的最大高度的最小值，下标 i 处能接的雨水量等于下标 i 处的水能到达的最大高度减去 height[i]。
　

朴素的做法是对于数组 height 中的每个元素，分别向左和向右扫描并记录左边和右边的最大高度，然后计算每个下标位置能接的雨水量。假设数组 height 的长度为 n，该做法需要对每个下标位置使用 O(n) 的时间向两边扫描并得到最大高度，因此总时间复杂度是 O(n^2)，但是**向左和向右扫描并记录左边和右边的最大高度**这一点很明显满足单调栈的特性，因此我们可以用单调栈来做。
  
维护一个单调栈，单调栈存储的是下标，满足从栈底到栈顶的下标对应的数组 height 中的元素递减。遍历柱子，当不满足当前遍历到的柱子小于栈顶柱子的高度时，说明栈顶右边第一个大于他的柱子出现了，然后栈顶出栈，此时的栈顶就是左边第一个大于出栈柱子的柱子，因此可以计算出栈柱子的接水面积了。

<img src="https://user-images.githubusercontent.com/28688510/155190481-f783337b-da43-4709-aacf-de62b04c574d.png" width="500">
<img src="https://user-images.githubusercontent.com/28688510/155190603-e5ecd101-058c-4142-9f1d-edd44ae11c60.png" width="500">

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

<img src="https://user-images.githubusercontent.com/28688510/155192585-5f6fc5b2-4f81-40f5-9db0-872b9fa0b847.png" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155357541-3de82267-d6f7-4ffb-a69d-24ca39166cf4.png" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155359599-5ca15f08-823b-4afb-9d53-39bbc540cbb1.png" width="500">
<img src="https://user-images.githubusercontent.com/28688510/155360290-8e5a5be7-2b14-40e9-8686-76ab381b7c14.gif" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155360915-793c2d2d-e8a2-4de6-b6d9-d293f341d538.png" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155362858-592060e5-fc95-4ec4-ba06-7ac67a2b1430.png" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155363919-504267ee-fc57-445d-981c-f38422ee3378.png" width="500">

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
------
<img src="https://user-images.githubusercontent.com/28688510/155365911-7025d885-3c2a-4b68-9048-00411a32a83a.png" width="500">

通过分别滑动A和B字符串，逐个比对元素是否相同，取相同部分长度返回，最终求最大值

```c++
class Solution {
public:
    int maxLength(vector<int>& A, vector<int>& B, int addA, int addB, int len) {
        int k = 0;
        int ans = 0;
        // 注意这里的len，不能越界
        for (int i = 0; i < len; i++) {
            if (A[addA + i] == B[addB + i]) {
                k++;
            }
            else {
                k = 0;
            }
            ans = max(ans, k);
        }
        return ans;
    }
    int findLength(vector<int>& A, vector<int>& B) {
        int sizeA = A.size(), sizeB = B.size();
        int ans = 0;
        // 滑动A后取最大的子串长度
        for (int i = 0; i < sizeA; i++) {
            int len = min(sizeB, sizeA - i);
            int tmp = maxLength(A, B, i, 0, len);
            ans = max(ans, tmp);
        }
        // 滑动B后取最大的子串长度
        for (int i = 0; i < sizeB; i++) {
            int len = min(sizeA, sizeB - i);
            int tmp = maxLength(A, B, 0, i, len);
            ans = max(ans, tmp);
        }
        return ans;
    }
};
```

## 双指针
双指针（Two Pointers）是这样一种模式：两个指针以一前一后的模式在数据结构中迭代，直到一个或两个指针达到某种特定条件。双指针通常在排序数组或链表中搜索配对时很有用；比如当你必须将一个数组的每个元素与其它元素做比较时。

双指针是很有用的，因为如果只有一个指针，你必须继续在数组中循环回来才能找到答案。这种使用单个迭代器进行来回在时间和空间复杂度上都很低效——这个概念被称为「渐进分析（asymptotic analysis）」。尽管使用 1 个指针进行暴力搜索或简单普通的解决方案也有效果，但这会让时间复杂度达到 O(n²) 。在很多情况下，双指针有助于你寻找有更好空间或运行时间复杂度的解决方案。

<img src="https://user-images.githubusercontent.com/28688510/155569214-f13083d9-63a2-4d87-8555-1b4cc2ed3024.png" width="500">

用于识别使用双指针的时机的方法：
* 可用于你要处理排序数组（或链接列表）并需要查找满足某些约束的一组元素的问题
* 数组中的元素集是配对、三元组甚至子数组

下面是一些满足双指针模式的问题：
* 求一个排序数组的平方（简单）
* 求总和为零的三元组（中等）
* 比较包含回退（backspace）的字符串（中等）
------
<img src="https://user-images.githubusercontent.com/28688510/155570731-08a6c857-71b3-4d56-b3d8-b5d01940f584.png" width="500">

显然，如果数组 nums 中的所有数都是非负数，那么将每个数平方后，数组仍然保持升序；如果数组 nums 中的所有数都是负数，那么将每个数平方后，数组会保持降序。通过找到分界点，给分界点和分界点左边一个作为两个数组的起始点（双指针），后采用归并排序的方法进行结果插入，可以省略排序的时间复杂度，使得时间复杂度达到O(n)。

```c++
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int n = nums.size();
        int right= 0;
        while (right < n && nums[right] < 0) {
            right++;
        }
        int left = right - 1;

        vector<int> res;
        while (left >= 0 || right < n) {
            int tmp = 0;
            if (left < 0) {
                tmp = nums[right] * nums[right];
                right++;
            }
            else if (right >= n) {
                tmp = nums[left] * nums[left];
                left--;
            }
            else {
                if (nums[left] * -1 <= nums[right]) {
                    tmp = nums[left] * nums[left];
                    left--;
                }
                else {
                    tmp = nums[right] * nums[right];
                    right++;
                }
            }
            res.push_back(tmp);
        }

        return res;
    }
};
```
------
<img src="https://user-images.githubusercontent.com/28688510/155574831-cecc046f-0c2c-4253-811a-a3c114365ac8.png" width="500">

这题很简单，因为是有序数组，双指针分别从首尾遍历即可，小于目标值说明左边应该变大，左指针右移，大于目标值说明右边应该变小，右指针左移

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int i = 0, j = numbers.size() - 1;
        while (i < j) {
            if (numbers[i] + numbers[j] == target) {
                return {i + 1, j + 1};
            }
            else if (numbers[i] + numbers[j] < target) {
                i++;
            }
            else {
                j--;
            }
        }
        return {};
    }
};
```
--------

那么如果是三数之和呢？同样，只需要固定住一个数，然后其余两个数按上面的方法确定即可。

<img src="https://user-images.githubusercontent.com/28688510/155573950-00a190c7-aa1b-4c15-af06-498a1ed7949e.png" width="500">

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int len = nums.size();
        vector<vector<int>> res;
        if (len < 3) return res;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < len - 2; i++) {
            int left = i + 1, right = len - 1, sum = 0 - nums[i];
            // 为了保证不加入重复的 list,因为是有序的，所以如果和前一个元素相同，只需要继续后移就可以
            if (i == 0 || i > 0 && (nums[i] != nums[i - 1])) {
                while (left < right) {
                    if (nums[left] + nums[right] == sum) {
                        res.push_back({nums[i], nums[left], nums[right]});
                        // 元素相同要后移，防止加入重复的list
                        while (left < right && nums[left] == nums[left + 1]) left++;
                        while (left < right && nums[right] == nums[right - 1]) right--;
                        left++; right--;
                    }
                    else if (nums[left] + nums[right] < sum) left++;
                    else right--;
                }
            }
        }
        return res;
    }
};
```
-------
如果是四数之和呢？与三个数解法类似，不同的是要用双重循环确定前两个数，同时要做一些剪枝操作：
* 在确定第一个数之后，如果 nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target，说明此时剩下的三个数无论取什么值，四数之和一定大于 target，因此退出第一重循环；
* 在确定第一个数之后，如果 nums[i]+nums[n−3]+nums[n−2]+nums[n−1]<target，说明此时剩下的三个数无论取什么值，四数之和一定小于 target，因此第一重循环直接进入下一轮，枚举 nums[i+1]；
* 在确定前两个数之后，如果 nums[i]+nums[j]+nums[j+1]+nums[j+2]>target，说明此时剩下的两个数无论取什么值，四数之和一定大于 target，因此退出第二重循环；
* 在确定前两个数之后，如果 nums[i]+nums[j]+nums[n−2]+nums[n−1]<target，说明此时剩下的两个数无论取什么值，四数之和一定小于 target，因此第二重循环直接进入下一轮，枚举 nums[j+1]。

<img src="https://user-images.githubusercontent.com/28688510/155582002-ba6dfff4-68df-428d-b8ce-d966a8f2df9a.png" width="500">

```c++
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());

        int n = nums.size();
        vector<vector<int>> res;

        for (int i = 0; i <= n - 4; i++) {
            // 避免重复结果
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            // 剪枝
            if ((long)nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target) break;
            if ((long)nums[i] + nums[n-1] + nums[n-2] + nums[n-3] < target) continue;
            
            for (int j = i + 1; j <= n - 3; j++) {
                // 避免重复结果
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                // 剪枝
                if ((long)nums[i] + nums[j] + nums[j+1] + nums[j+2] > target) break;
                if ((long)nums[i] + nums[j] + nums[n-1] + nums[n-2] < target) continue;

                int sum = target - nums[i] - nums[j];
                int left = j + 1, right = n - 1;

                while (left < right) {
                    if (nums[left] + nums[right] == sum) {
                        res.push_back({nums[i], nums[j], nums[left], nums[right]});
                        // 避免出现重复结果，跳过相同的部分
                        // 由于循环是找到最后一个相同数字的位置，因此跳出循环后还要右移一次
                        while (left < right && nums[left] == nums[left+1]) {
                            left++;
                        }
                        left++;
                        // 避免出现重复结果，跳过相同的部分
                        // 由于循环是找到最后一个相同数字的位置，因此跳出循环后还要左移一次
                        while (left < right && nums[right] == nums[right-1]) {
                            right--;
                        }
                        right--;
                    }
                    else if (nums[left] + nums[right] > sum) {
                        right--;
                    }
                    else {
                        left++;
                    }
                }
            }
        }

        return res;
    }
};
```

------
<img src="https://user-images.githubusercontent.com/28688510/155875993-dbbd8f74-eda8-488b-b21d-548400e9958b.png" width="500">


这一题可以用栈很简单地解决：
```c++
class Solution {
public:
    bool backspaceCompare(string S, string T) {
        return build(S) == build(T);
    }

    string build(string str) {
        string ret;
        for (char ch : str) {
            if (ch != '#') {
                ret.push_back(ch);
            } else if (!ret.empty()) {
                ret.pop_back();
            }
        }
        return ret;
    }
};
```

当然，也可以使用双指针：（**比较不直观，建议还是用栈做，时间复杂度都是O(M+N)**）
一个字符是否会被删掉，只取决于该字符后面的退格符，而与该字符前面的退格符无关。因此当我们逆序地遍历字符串，就可以立即确定当前字符是否会被删掉。我们定义两个指针，分别指向两字符串的末尾。每次我们让两指针逆序地遍历两字符串，直到两字符串能够各自确定一个字符，然后将这两个字符进行比较。重复这一过程直到找到的两个字符不相等，或遍历完字符串为止。

```c++
class Solution {
public:
    bool backspaceCompare(string S, string T) {
        int i = S.length() - 1, j = T.length() - 1;
        int skipS = 0, skipT = 0;

        while (i >= 0 || j >= 0) {
            while (i >= 0) {
                if (S[i] == '#') {
                    skipS++, i--;
                } else if (skipS > 0) {
                    skipS--, i--;
                } else {
                    break;
                }
            }
            while (j >= 0) {
                if (T[j] == '#') {
                    skipT++, j--;
                } else if (skipT > 0) {
                    skipT--, j--;
                } else {
                    break;
                }
            }
            if (i >= 0 && j >= 0) {
                if (S[i] != T[j]) {
                    return false;
                }
            } else {
                if (i >= 0 || j >= 0) {
                    return false;
                }
            }
            i--, j--;
        }
        return true;
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/155876346-014941f9-90c9-499b-8039-252219568612.png" width="500">

典型的双指针题，左右分别向中间靠拢，哪边小哪边移动，移动的同时计算盛水的容量，最终取最大值

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int len = height.size();
        int res = INT_MIN;
        int left = 0, right = len - 1;

        while (left < right) {
            res = max(res, min(height[left], height[right]) * (right - left));
            if (height[left] > height[right]) right--;
            else left++;
        }

        return res;
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/155876749-aebce8a6-d223-4581-a2fe-26e2b0f59bc9.png" width="500">

本题是经典的「荷兰国旗问题」，可以统计出数组中 0, 1, 2 的个数，再根据它们的数量，重写整个数组。这种方法较为简单，也很容易想到。
通过双指针方法，可以达到一遍遍历完成排序的效果。

双指针设立在两头，当遍历到0时，往前交换，左指针右移，当遍历到2时，往后交换，右指针左移，当遍历到右指针位置时，结束遍历。
```c++
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int p0 = 0, p2 = n - 1;
        for (int i = 0; i <= p2; ++i) {
            while (i <= p2 && nums[i] == 2) {
                swap(nums[i], nums[p2]);
                --p2;
            }
            if (nums[i] == 0) {
                swap(nums[i], nums[p0]);
                ++p0;
            }
        }
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156201706-fc251cab-1f68-4185-ac22-893b3f439743.png" width="500">

这一题非常经典，如果不用双指针的话，需要两次遍历，第一次求整个链表的长度，第二次再到指定位置去删除节点。
如果是使用双指针的话，让快指针先走n步，然后快慢指针再同时走，当快指针走到末尾结点时，慢指针即找到待删除节点。**这里要考虑只有一个结点的情况，我们将一个新的空结点指向头结点，就可以避免这种情况而不去特殊处理，最终返回这个新结点的下一个结点作为结果，也就避免了头结点被删除的情况**

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 设置结点指向头结点
        ListNode* pre = new ListNode();
        pre->next = head;

        ListNode* slow = pre;
        ListNode* fast = pre;

        // 快指针先走
        for (int i = 0; i < n; i++) {
            fast = fast->next;
        }

        // 快慢指针同时走，直到快指针到最后
        while (fast->next) {
            slow = slow->next;
            fast = fast->next;
        }

        // 此时慢指针即是待删除的结点的前一个结点
        slow->next = slow->next->next;

        return pre->next;
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156203495-ff5b6161-d6b5-493a-9425-72740f5914cb.png" width="500">

这个题如果没做过还是相当有难度的，这里直接贴一下官方题解的思路

<img src="https://user-images.githubusercontent.com/28688510/156203694-5df9b100-45c6-4463-9e72-ff99d5fad06e.png" width="600">

![31](https://user-images.githubusercontent.com/28688510/156204668-211b7e7a-4673-4812-993f-af24cc1bc56f.gif)

好像与双指针没太大关系哈。。。

```c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i = nums.size() - 2;
        // 从后往前找到第一个nums[i] < nums[i + 1]的元素
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        
        // 如果i<0则表明整个数组都降序排列
        if (i >= 0) {
            int j = nums.size() - 1;
            // 在区间[i+1：]找到第一个大于nums[i]的元素
            // 这里没有令j >= i+1，因为区间[i+1：]降序排列，所以一定能在这个区间内找到大于nums[i]的元素
            while (j >= 0 && nums[i] >= nums[j]) {
                j--;
            }
            // 交换这两个数
            swap(nums[i], nums[j]);
        }
        // 反转降序的部分
        reverse(nums.begin() + i + 1, nums.end());
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156209007-308f57e7-097c-4f66-a791-9c4b40841796.png" width="500">

这个题目有两种思路：

1. 二分法：

<img src="https://user-images.githubusercontent.com/28688510/156209366-aab7736c-c975-4157-89f8-f154c95133e2.png" width="600">

时间复杂度：O(nlogn)，其中 nn 为 nums 数组的长度。O(logn) 代表了我们枚举二进制数的位数个数，枚举第 i 位的时候需要遍历数组统计 x 和 y 的答案，因此总时间复杂度为 O(nlogn)。

```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        int l = 1, r = n - 1, ans = -1;
        while (l <= r) {
            int mid = (l + r) >> 1;
            
            // 统计小于等于mid的个数
            int cnt = 0;
            for (int i = 0; i < n; ++i) {
                cnt += nums[i] <= mid;
            }
            
            if (cnt <= mid) {
                l = mid + 1;
            } else {
                r = mid - 1;
                ans = mid;
            }
        }
        return ans;
    }
};
```

2. 双指针（快慢）：

这里就依赖于对 「Floyd 判圈算法」（又称龟兔赛跑算法）有所了解，它是一个检测链表是否有环的算法，LeetCode 中相关例题有 141. 环形链表，142. 环形链表 II。

我们对 nums 数组建图，每个位置 i 连一条 i→nums[i] 的边。由于存在的重复的数字 target，因此 target 这个位置一定有起码两条指向它的边，因此整张图一定存在环，且我们要找到的 target 就是这个环的入口，那么整个问题就等价于 142. 环形链表 II。

**我们先设置慢指针 slow 和快指针 fast ，慢指针每次走一步，快指针每次走两步，根据「Floyd 判圈算法」两个指针在有环的情况下一定会相遇，此时我们再将 slow 放置起点 0，两个指针每次同时移动一步，相遇的点就是答案。**

<img src="https://user-images.githubusercontent.com/28688510/156212218-043d02c1-c9e8-40af-8578-63f387d93e7b.png" width="600">

```c++
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = 0, fast = 0;
        // 从头开始，慢指针每次走一步，快指针每次走两步直到相遇
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        
        // 将慢指针设回开头位置
        slow = 0;
        // 此时再将两个指针每次同时移动一步，相遇的点就是环入口
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156330013-1a8c3c64-2d1b-40dd-b2e6-48707bb71cf5.png" width="500">

这个题目通过**快慢指针实现二分**，然后将两部分进行归并排序，很巧妙

```c++
class Solution {
public:
    ListNode* mergeSort(ListNode* head1, ListNode* head2) {
        ListNode* start = new ListNode();
        ListNode* ans = start;

        while (head1 && head2) {
            if (head1->val < head2->val) {
                start->next = head1;
                start = start->next;
                head1 = head1->next;
            }
            else {
                start->next = head2;
                start = start->next;
                head2 = head2->next;
            }
        }
        if (!head1) {
            while (head2) {
                start->next = head2;
                start = start->next;
                head2 = head2->next;
            }
        }
        if (!head2) {
            while (head1) {
                start->next = head1;
                start = start->next;
                head1 = head1->next;
            }
        }
        return ans->next;
    }

    ListNode* sortList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return head;
        ListNode* start = new ListNode();
        start->next = head;
        ListNode* slow = start;
        ListNode* fast = start;

        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode* right_head = slow->next;
        slow->next = nullptr;

        ListNode* h1 = sortList(head);
        ListNode* h2 = sortList(right_head);
        return mergeSort(h1, h2);
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156335833-ee12edee-6642-491d-91b7-fd98d5f93deb.png" width="500">

又是很巧妙的一种双指针解法，当一条路径遍历完后让其从另一条路径开始，这样会让第二次遍历时在交点处相遇。

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* a = headA;
        ListNode* b = headB;

        while (a != b) {
            a = a == nullptr? headB: a->next;
            b = b == nullptr? headA: b->next;
        }
        return a;
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156338047-314b3b27-11d7-4681-a9cc-f8648dc7f77e.png" width="500">

遍历到每个单词的起止位置，用双指针反转字符串即可。

```c++
class Solution {
public: 
    string reverseWords(string s) {
        int length = s.length();
        int i = 0;
        while (i < length) {
            int start = i;
            // 遍历到单词结束的空格位置
            while (i < length && s[i] != ' ') {
                i++;
            }

            // 反转字符串
            int left = start, right = i - 1;
            while (left < right) {
                swap(s[left], s[right]);
                left++;
                right--;
            }
            
            // 将单词后面的空格走完
            while (i < length && s[i] == ' ') {
                i++;
            }
        }
        return s;
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156394514-d2e8d9af-ba9a-4cae-ba8c-7516a87aec38.png" width="500">

很经典的一道题，如果想要空间复杂度为O(1)的话，则需要用的双指针算法（快慢指针），具体流程为：

* 快慢指针找到中点
* 以中点为界分成两个链表
* 反转其中一个链表
* 双指针依次判断这两个链表是否一一对应相等

```c++
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        // 快慢指针找中点
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }
        
        // 反转链表
        ListNode* new_head = reverseList(slow);
        
        // 一一对应是否相等
        while (head && new_head) {
            if (head->val != new_head->val) {
                return false;
            }
            head = head->next;
            new_head = new_head->next;
        }
        return true;
    }

    ListNode* reverseList(ListNode* head) {
        if (head == nullptr) return head;
        ListNode* pre = nullptr;
        while (head) {
            ListNode* tmp = head->next;
            head->next = pre;
            pre = head;
            head = tmp;
        }
        return pre;
    }
};
```

------

<img src="https://user-images.githubusercontent.com/28688510/156400051-fcbd7eda-f096-4ee6-8f04-b6df09b111f9.png" width="500">

这是快慢指针十分典型的题目，循环数组类型，但由于题目条件较为复杂，要求循环路径只能同向，因此需要进行一些限制

```c++
class Solution {
public:
    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size();
        auto next = [&](int i) {
            return ((i + nums[i]) % n + n) % n;
        };

        for (int i = 0; i < n; i++) {
            int slow = i;
            int fast = next(i);

            // 每一步都要满足题目的要求，循环路径同向
            // 注意，这里如果不加nums[slow] * nums[next(fast)] > 0
            // 由于快指针一次走两步，可能会导致某些异号的情况漏掉，具体见下面评论
            // 或者写成nums[slow]*nums[next(slow)] > 0 && nums[fast]*nums[next(fast)] > 0也可行
            while (nums[slow] * nums[fast] > 0 && nums[slow] * nums[next(fast)] > 0) {
                // 快慢指针相遇，说明存在环
                if (slow == fast) {
                    // 如果slow == next(slow)，说明环的长度为1
                    if (slow == next(slow)) break;
                    // 否则说明找到环了
                    else return true;
                }
                // 快慢指针移动
                slow = next(slow);
                fast = next(next(fast));
            }

            // 当没找到环并且轨迹反向时，说明i作为起点这条路不通
            // 可以将这条路上同向的部分设置成0，保证后面不再走这条同向的路
            // 起到了剪枝的效果
            int tmp = i;
            while (nums[tmp] * nums[next(tmp)] > 0) {
                int num = tmp;
                tmp = next(num);
                nums[num] = 0;
            } 
        }

        return false;
    }
};
```

这里贴两个解释，为什么循环条件要加上nums[slow] * nums[next(fast)] > 0：

<img src="https://user-images.githubusercontent.com/28688510/156629077-97cbe737-a51f-427b-83e6-e2491ac6ffef.png" width="600">

<img src="https://user-images.githubusercontent.com/28688510/156629222-e50f77e6-e148-49f1-9d7d-034c33cf93e0.png" width="600">


## 合并区间

合并区间模式是一种处理重叠区间的有效技术。在很多涉及区间的问题中，你既需要找到重叠的区间，也需要在这些区间重叠时合并它们。该模式的工作方式为：

给定两个区间（a 和 b），这两个区间有 6 种不同的互相关联的方式：

理解并识别这六种情况有助于你求解范围广泛的问题，从插入区间到优化区间合并等。

那么如何确定何时该使用合并区间模式呢？

如果你被要求得到一个仅含互斥区间的列表

如果你听到了术语「重叠区间（overlapping intervals）」

合并区间模式的问题：

* 区间交叉（中等）

* 最大 CPU 负载（困难）


<img width="638" alt="image" src="https://user-images.githubusercontent.com/28688510/157081528-d2484162-e310-49ce-b897-17e93d93634d.png">

注意要先按区间头点排序，再合并

```c++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.size() == 0) {
            return {};
        }
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> merged;
        for (int i = 0; i < intervals.size(); ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            if (!merged.size() || merged.back()[1] < L) {
                merged.push_back({L, R});
            }
            else {
                merged.back()[1] = max(merged.back()[1], R);
            }
        }
        return merged;
    }
};
```

------

<img width="642" alt="image" src="https://user-images.githubusercontent.com/28688510/157082064-d3fbb8bf-41f2-4bd6-a719-c35689c90056.png">

这里需要注意处理插入的条件，还有最后的部分，如果没有插入要记得插入

```c++
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        int n = intervals.size();

        vector<vector<int>> res;
        if (n == 0) return {newInterval};

        int left = newInterval[0], right = newInterval[1];
        
        bool placed = false; // 记录新区间有没有插入
        for (int i = 0; i < n; i++) {
            // 由于intervals是升序排列
            // 当intervals[i][0] > right时，说明后面都比right大
            // 如果没插入left right, 此时应该插入left right
            // 否则插入intervals[i]
            if (intervals[i][0] > right) {
                if (!placed) {
                    res.push_back({left, right});
                    placed = true;
                }
                res.push_back(intervals[i]);
            }

            if (intervals[i][1] < left) {
                res.push_back(intervals[i]);
            }
            else {
                left = min(intervals[i][0], left);
                right = max(intervals[i][1], right);
            }
        }
        // 这里不能忘记，如果遍历到最后都没有插入leftright，需要插入
        if (!placed) {
            res.push_back({left, right});
        }

        return res;
    }
};
```

## 堆

<img width="525" alt="image" src="https://user-images.githubusercontent.com/28688510/157095606-a8b14e41-e60b-428b-86f4-2f44107b88b0.png">

```c++
class Solution {
public:
    vector<int> getOrder(vector<vector<int>>& tasks) {
        int n = tasks.size();
        // 创建小顶堆
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
        // index用于存储排序后的task索引位置
        vector<int> index(n);

        // 从0开始逐步加1赋初始值
        iota(index.begin(), index.end(), 0);

        // 排序
        sort(index.begin(), index.end(), [&](int i, int j) {
            return tasks[i][0] < tasks[j][0];
        });

        vector<int> res;
        // 下次任务要执行的时间
        long long t = 0; 
        // 遍历index的指针
        int ptr = 0;

        while (res.size() != n) {
            // 当堆为空时，说明没有待执行的任务，快进到下一次可执行的任务
            if (q.empty()) {
                t = max(t, (long long)tasks[index[ptr]][0]);
            }
            
            // 当有任务执行时，将本次任务执行完前的其他任务入堆
            while (ptr < n && tasks[index[ptr]][0] <= t) {
                q.push(make_pair(tasks[index[ptr]][1], index[ptr]));
                ptr++;
            }

            // 选择处理时间最小的任务
            t += q.top().first;
            res.push_back(q.top().second);
            q.pop();
        }

        return res;
    }
};
```

## 二分法

在有序的数组或者链表中，找到某个分界点，可以套用二分法，将时间复杂度降低到O(logN)
注意二分条件，最终结果会落到left+1，要取分界点为左边均不满足结果，右边可以满足结果

<img width="525" alt="image" src="https://user-images.githubusercontent.com/28688510/157299667-271dfbe0-4111-4ae4-8176-d21c609f6d3b.png">

```c++
class Solution {
public:
    int findKthPositive(vector<int>& arr, int k) {
        int n = arr.size();
        int left = 0, right = n - 1;

        while (left <= right) {
            int mid = (left + right) / 2;
            // 可知arr[mid] - mid - 1就是arr[mid]处缺失的整数个数
            // 这里的目的是找到第一个大于或等于K的位置
            if (arr[mid] - mid - 1 < k)
                left = mid + 1;
            else 
                right = mid - 1;
        }

        // 化简 arr[left - 1] + k - (arr[left - 1] - (left - 1) - 1) 而来
        // 可以避免left = 0时出错
        return k + left;
    }
};
```

------

<img width="525" alt="image" src="https://user-images.githubusercontent.com/28688510/157303511-2d056d0a-58a6-493a-b701-afc0fc17fb39.png">

![image](https://user-images.githubusercontent.com/28688510/157303883-9c21c5af-ec32-44d7-bd88-6ff591ac5a7f.png)

![image](https://user-images.githubusercontent.com/28688510/157303938-4b1ceb92-1375-46ea-9b98-1ed86ed12e0e.png)


```c++
class Solution {
public:
    bool check(vector<vector<int>>& matrix, int k, int mid, int n) {
        int num = 0;
        int i = n - 1;
        int j = 0;
        // 沿着边界走，同时统计左上角矩阵元素的个数
        while (i >= 0 && j < n) {
            // 往右走
            if (matrix[i][j] <= mid) {
                num += i + 1; // 注意这里右走一格加一竖列的个数
                j++;
            }
            // 往上走
            else {
                // num += j;
                i--;
            }
        }
        // cout << mid << " | " << num << endl;
        return num < k;  // 这里不能带等号，带了会导致取mid无限循环
    }
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = matrix.size();
        int left = matrix[0][0];
        int right = matrix[n-1][n-1];
        while (left < right) {
            // cout << left << " and " << right << endl;
            int mid = (left + right) / 2;
            // mid为界限划分矩阵，左上角小于等于mid，右下角大于mid
            // 统计左上角个数，如果小于k，说明第k小的数大于mid
            // 否则第k小的数小于等于mid
            if (check(matrix, k, mid, n)) {
                left = mid + 1;
            }
            else {
                right = mid;
            } 
        }
        return left;
    }
};
```

## 树

### 深度优先遍历DFS
Tree DFS 是基于深度优先搜索（DFS）技术来遍历树。

你可以使用递归（或该迭代方法的技术栈）来在遍历期间保持对所有之前的（父）节点的跟踪。

Tree DFS 模式的工作方式是从树的根部开始，如果这个节点不是一个叶节点，则需要做三件事：

1．决定现在是处理当前的节点（pre-order），或是在处理两个子节点之间（in-order），还是在处理两个子节点之后（post-order）
2. 为当前节点的两个子节点执行两次递归调用以处理它们

如何识别 Tree DFS 模式：

* 如果你被要求用 in-order、pre-order 或 post-order DFS 来遍历一个树
* 如果问题需要搜索其中节点更接近叶节点的东西

Tree DFS 模式的问题：

* 路径数量之和（中等）
* 一个和的所有路径（中等）

<img width="525" alt="image" src="https://user-images.githubusercontent.com/28688510/157497174-25f7926e-f14b-4765-8809-1bf07c641b43.png">

1. 递归版本很简单：

```c++
class Solution {
public:
    void inorder(TreeNode* root, vector<int>& res) {
        if (!root) {
            return;
        }
        inorder(root->left, res);
        res.push_back(root->val);
        inorder(root->right, res);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        inorder(root, res);
        return res;
    }
```

2. 迭代版本（使用栈）相对复杂一点：

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
        while (root != nullptr || !stk.empty()) {
            while (root != nullptr) {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            res.push_back(root->val);
            root = root->right;
        }
        return res;
    }
};
```

------

后序遍历对于迭代算法来说更加复杂一些，需要记录前一个输出的节点，并判断右子树的状态来确定是输出当前结点还是先遍历右子树。

```c++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        // 迭代写法
        vector<int> res;
        // 用于记录上一个加入res的节点
        TreeNode* pre = nullptr;
        stack<TreeNode*> stk;

        while (root != nullptr || !stk.empty()) {
            // 往左孩子走到底
            while (root != nullptr) {
                stk.emplace(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            // 判断最左节点的右孩子是否为空，若为空，则说明需要输出
            // 或者存在右孩子且已经输出完了，输出当前节点
            if (root->right == nullptr || root->right == pre) {
                res.emplace_back(root->val);
                // 记录当前输出的节点
                pre = root;
                // 将当前节点置为空，方便下一次循环取栈顶节点
                root = nullptr;
            }
            // 如果当前节点右孩子不为空，且还没有输出右子树部分
            // 则将当前节点重新进栈，右子树按后序遍历进栈
            else {
                stk.emplace(root);
                root = root->right;
            }
        }

        return res;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/157081330-44df3d5d-b5c7-42b9-b03d-0275701ec86b.png">

```c++
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        if (A == nullptr || B == nullptr) return false;
        // 先序遍历时，首先判断以A为根节点的子树是否包含B
        // 如果是，后面就不用继续递归了，所以使用||
        return recur(A, B) || isSubStructure(A->left, B) || isSubStructure(A->right, B);
    }

    // 此函数用于判断以A为根节点的子树是否包含B
    bool recur(TreeNode* A, TreeNode* B) {
        // 如果B遍历到空指针了，说明前面都相等了，返回true
        if (B == nullptr) return true;
        // 如果AB的值不同，说明是不同的子结构
        // 注意，这里需要添加A是否是空指针的判断
        if (A == nullptr || A->val != B->val) return false;
        // 如果AB值相同，继续递归判断其左右子树
        return recur(A->left, B->left) && recur(A->right, B->right);
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160285014-61a629f1-8173-4c43-8d0d-8da01cfe4578.png">

1. DFS递归

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160285251-64a5bbd0-4654-4fd5-8abd-2c3485f07270.png">
<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160285285-ba51ab64-b4ef-4b1f-994b-449c63cb3e0b.png">

```c++
class Solution {
public:
    bool dfs(TreeNode* root, const long& low, const long& high) {
        if (!root) return true;
        if (root->val <= low || root->val >= high) return false;
        return dfs(root->left, low, root->val) && dfs(root->right, root->val, high);
    }
    bool isValidBST(TreeNode* root) {
        return dfs(root, LONG_MIN, LONG_MAX);
    }
};
```

2. 中序遍历

根据二叉搜索树的性质可知，中序遍历的结果一定是升序的，如果出现了上一个遍历到的值大于当前遍历到的值，则说明不是AVL

```c++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        stack<TreeNode*> stack;
        long long inorder = (long long)INT_MIN - 1;

        while (!stack.empty() || root != nullptr) {
            while (root != nullptr) {
                stack.push(root);
                root = root -> left;
            }
            root = stack.top();
            stack.pop();
            // 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
            if (root -> val <= inorder) {
                return false;
            }
            inorder = root -> val;
            root = root -> right;
        }
        return true;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160285663-6aa3fc07-d3d5-420e-b4c7-a49686441fda.png">

与上一题相同，二叉搜索树中序遍历结果一定是递增的，如果有两个数被交换，那么一定存在非递增的部分：
注意如果是中序遍历相邻的两个节点交换位置的话，只有一个数ai > ai+1，否则会有两个数ai > ai+1，aj > aj+1

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160285733-df7e7e50-5920-4ed7-b9ab-5d4aaac2c7cb.png">

```c++
class Solution {
public:
    void recoverTree(TreeNode* root) {
        stack<TreeNode*> stk;
        TreeNode* x = nullptr;
        TreeNode* y = nullptr;
        TreeNode* pred = nullptr;

        while (!stk.empty() || root != nullptr) {
            while (root != nullptr) {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            // 当root值小于前一个值时，说明这个数有问题
            if (pred != nullptr && root->val < pred->val) {
                // 因为如果存在不相邻的两个位置时，是要记录i和j+1的
                
                // 用y记录这个值
                y = root;
                // 如果之前x没有记录的话，用x记录前一个值
                if (x == nullptr) {
                    x = pred;
                }
                else break;
            }
            pred = root;
            root = root->right;
        }

        swap(x->val, y->val);
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160287395-223158ab-38da-49b0-bcc7-17a64358de4a.png">

1. DFS

```c++
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) {
            return true;
        } else if (p == nullptr || q == nullptr) {
            return false;
        } else if (p->val != q->val) {
            return false;
        } else {
            return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        }
    }
};
```

2. BFS

异或^ ：相同是0，不同是1

```c++
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (p == nullptr && q == nullptr) {
            return true;
        } else if (p == nullptr || q == nullptr) {
            return false;
        }
        queue <TreeNode*> queue1, queue2;
        queue1.push(p);
        queue2.push(q);
        while (!queue1.empty() && !queue2.empty()) {
            auto node1 = queue1.front();
            queue1.pop();
            auto node2 = queue2.front();
            queue2.pop();
            if (node1->val != node2->val) {
                return false;
            }
            auto left1 = node1->left, right1 = node1->right, left2 = node2->left, right2 = node2->right;
            if ((left1 == nullptr) ^ (left2 == nullptr)) {
                return false;
            }
            if ((right1 == nullptr) ^ (right2 == nullptr)) {
                return false;
            }
            if (left1 != nullptr) {
                queue1.push(left1);
            }
            if (right1 != nullptr) {
                queue1.push(right1);
            }
            if (left2 != nullptr) {
                queue2.push(left2);
            }
            if (right2 != nullptr) {
                queue2.push(right2);
            }
        }
        return queue1.empty() && queue2.empty();
    }
};
```

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160287746-8dbd82d4-383b-41b7-871c-8f5d5d904b2f.png">

1. DFS（递归）

```c++
class Solution {
public:
    bool dfs(TreeNode* left, TreeNode* right) {
        if (!left && !right) return true;
        if (!left || !right) return false;
        return left->val == right->val && dfs(left->left, right->right) && dfs(left->right, right->left);
    }
    bool isSymmetric(TreeNode* root) {
        if (!root) return false;
        return dfs(root, root);
    }
};
```

2. BFS（迭代）

```c++
class Solution {
public:
    bool check(TreeNode *u, TreeNode *v) {
        queue <TreeNode*> q;
        q.push(u); q.push(v);
        while (!q.empty()) {
            u = q.front(); q.pop();
            v = q.front(); q.pop();
            if (!u && !v) continue;
            if ((!u || !v) || (u->val != v->val)) return false;

            q.push(u->left); 
            q.push(v->right);

            q.push(u->right); 
            q.push(v->left);
        }
        return true;
    }

    bool isSymmetric(TreeNode* root) {
        return check(root, root);
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160288338-2d2ff656-8930-4565-b03e-c967815fdfc2.png">

1. DFS

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```

2. BFS

这种做法相对繁琐一些，需要遍历的时候在内层给一个循环控制输出完一层的结点后再进行下一次的大循环

```c++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        queue<TreeNode*> Q;
        Q.push(root);
        int ans = 0;
        while (!Q.empty()) {
            // 控制遍历一层的结点
            int sz = Q.size();
            while (sz > 0) {
                TreeNode* node = Q.front();Q.pop();
                if (node->left) Q.push(node->left);
                if (node->right) Q.push(node->right);
                sz -= 1;
            }
            ans += 1;
        } 
        return ans;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160288541-150c2dc9-79a1-4f9e-bf16-27a36ac0077f.png">

1. DFS（自顶向下）

这里与上一题的方法相似，先求左右子树的深度，然后判断是否相差大于1，同时左右子树是否为平衡二叉树

```c++
class Solution {
public:
    int height(TreeNode* root) {
        if (root == NULL) {
            return 0;
        } else {
            return max(height(root->left), height(root->right)) + 1;
        }
    }

    bool isBalanced(TreeNode* root) {
        if (root == NULL) {
            return true;
        } else {
            return abs(height(root->left) - height(root->right)) <= 1 && isBalanced(root->left) && isBalanced(root->right);
        }
    }
};
```

2. DFS（自底向上）

方法一由于是自顶向下递归，因此对于同一个节点，函数 height 会被重复调用，导致时间复杂度较高。如果使用自底向上的做法，则对于每个节点，函数 height 只会被调用一次。

自底向上递归的做法类似于后序遍历，对于当前遍历到的节点，先递归地判断其左右子树是否平衡，再判断以当前节点为根的子树是否平衡。如果一棵子树是平衡的，则返回其高度（高度一定是非负整数），否则返回−1。如果存在一棵子树不平衡，则整个二叉树一定不平衡。

```c++
class Solution {
public:
    int height(TreeNode* root) {
        if (root == NULL) {
            return 0;
        }
        int leftHeight = height(root->left);
        int rightHeight = height(root->right);
        if (leftHeight == -1 || rightHeight == -1 || abs(leftHeight - rightHeight) > 1) {
            return -1;
        } else {
            return max(leftHeight, rightHeight) + 1;
        }
    }

    bool isBalanced(TreeNode* root) {
        return height(root) >= 0;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160288967-cc51d05a-71d2-4782-81db-b98067339418.png">

1. DFS

```c++
class Solution {
public:
    int minDepth(TreeNode *root) {
        if (root == nullptr) {
            return 0;
        }
        // 当左右子树都为空时，最小深度为1
        if (root->left == nullptr && root->right == nullptr) {
            return 1;
        }

        // 最小深度为左右子树的最小深度的最小值 + 1
        int min_depth = INT_MAX;
        if (root->left != nullptr) {
            min_depth = min(minDepth(root->left), min_depth);
        }
        if (root->right != nullptr) {
            min_depth = min(minDepth(root->right), min_depth);
        }

        return min_depth + 1;
    }
};
```

2. BFS

```c++
class Solution {
public:
    int minDepth(TreeNode *root) {
        if (root == nullptr) {
            return 0;
        }

        queue<pair<TreeNode *, int> > que;
        que.emplace(root, 1);
        while (!que.empty()) {
            TreeNode *node = que.front().first;
            int depth = que.front().second;
            que.pop();
            if (node->left == nullptr && node->right == nullptr) {
                return depth;
            }
            if (node->left != nullptr) {
                que.emplace(node->left, depth + 1);
            }
            if (node->right != nullptr) {
                que.emplace(node->right, depth + 1);
            }
        }

        return 0;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160289330-8b158e7e-4ed7-40f5-9d04-cba3251c9731.png">

1. DFS

```c++
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if (!root) {
            return false;
        }
        // 为叶子节点且路径和为sum时
        if (!(root->left || root->right) && (root->val == sum)) {
            return true;
        }
        // 否则要找左孩子或右孩子节点以下是否有存在和为sum-root->val的路径
        else {
            return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
        }
    }
};
```

2. BFS

广度优先搜索比较繁琐一点，需要两个队列，一个记录遍历的结点，一个记录该节点的路径和

```c++
class Solution {
public:
    bool hasPathSum(TreeNode *root, int sum) {
        if (root == nullptr) {
            return false;
        }
        queue<TreeNode *> que_node;
        queue<int> que_val;
        que_node.push(root);
        que_val.push(root->val);
        while (!que_node.empty()) {
            TreeNode *now = que_node.front();
            int temp = que_val.front();
            que_node.pop();
            que_val.pop();
            if (now->left == nullptr && now->right == nullptr) {
                if (temp == sum) {
                    return true;
                }
                continue;
            }
            if (now->left != nullptr) {
                que_node.push(now->left);
                que_val.push(now->left->val + temp);
            }
            if (now->right != nullptr) {
                que_node.push(now->right);
                que_val.push(now->right->val + temp);
            }
        }
        return false;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160289916-67943370-b9e3-4579-a904-fa9689376bf2.png">

上一题的进化版，其实难度差不多，同样可以用DFS和BFS做

1. DFS

```c++
class Solution {
private:
    vector<vector<int>> res;
public:
    void dfs(TreeNode* root, int targetSum, vector<int> tmp) {
        if (!root) return;
        tmp.push_back(root->val);
        if (!(root->left || root->right) && root->val == targetSum) {
            res.push_back(tmp);
            return;
        }
        dfs(root->left, targetSum - root->val, tmp);
        dfs(root->right, targetSum - root->val, tmp);
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        vector<int> tmp;
        dfs(root, targetSum, tmp);
        return res;
    }
};
```

2. BFS

BFS更繁琐了，还需要用一个哈希表去记录父节点，然后得到整体的路径，不建议用BFS做

```c++
class Solution {
public:
    vector<vector<int>> ret;
    // 记录父节点
    unordered_map<TreeNode*, TreeNode*> parent;
    
    // 不断往上得到父节点，最后反转得到路径
    void getPath(TreeNode* node) {
        vector<int> tmp;
        while (node != nullptr) {
            tmp.emplace_back(node->val);
            node = parent[node];
        }
        reverse(tmp.begin(), tmp.end());
        ret.emplace_back(tmp);
    }

    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) {
            return ret;
        }
        
        // BFS遍历所用的队列
        queue<TreeNode*> que_node;
        // 记录对应遍历位置的路径和
        queue<int> que_sum;
        que_node.emplace(root);
        que_sum.emplace(0);

        while (!que_node.empty()) {
            TreeNode* node = que_node.front();
            que_node.pop();
            int rec = que_sum.front() + node->val;
            que_sum.pop();

            if (node->left == nullptr && node->right == nullptr) {
                if (rec == targetSum) {
                    getPath(node);
                }
            } else {
                if (node->left != nullptr) {
                    // 更新父节点的字典
                    parent[node->left] = node;
                    que_node.emplace(node->left);
                    que_sum.emplace(rec);
                }
                if (node->right != nullptr) {
                    // 更新父节点的字典
                    parent[node->right] = node;
                    que_node.emplace(node->right);
                    que_sum.emplace(rec);
                }
            }
        }

        return ret;
    }
};
```
------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160290811-33d18856-6862-4ae3-a4cf-c7ca5c480e5a.png">

这个思路很牛逼，建议多看，相当于倒序的前序遍历，这样就能得到前序遍历的后一个节点，然后让当前节点去指向他。

```c++
class Solution {
public:
    TreeNode* last = nullptr;
    void flatten(TreeNode* root) {
        if (root == nullptr) return;
        flatten(root->right);
        flatten(root->left);
        root->right = last;
        root->left = nullptr;
        last = root;
    }
};
```

或者采用更容易理解的前序遍历+构造的形式去做

```c++
class Solution {
public:
    void flatten(TreeNode* root) {
        vector<TreeNode*> l;
        preorderTraversal(root, l);
        int n = l.size();
        for (int i = 1; i < n; i++) {
            TreeNode *prev = l.at(i - 1), *curr = l.at(i);
            prev->left = nullptr;
            prev->right = curr;
        }
    }

    void preorderTraversal(TreeNode* root, vector<TreeNode*> &l) {
        if (root != NULL) {
            l.push_back(root);
            preorderTraversal(root->left, l);
            preorderTraversal(root->right, l);
        }
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160428506-c24487da-03f2-4994-8220-bc44f099b2ba.png">

1. 用BFS可以很简单地做出来

```c++
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        queue<Node*> q;
        q.push(root);
        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                Node* tmp = q.front();
                q.pop();
                tmp->next = (i == size-1)? NULL: q.front();
                if (tmp->left)
                    q.push(tmp->left);
                if (tmp->right)
                    q.push(tmp->right);
            }
        }
        return root;
    }
};
```

2. 如果面试官变态，要你用O(1)空间做，就把下面的代码呼他脸上

<img width="400" alt="image" src="https://user-images.githubusercontent.com/28688510/160430597-81b25af1-8ebd-4423-8685-c7923e53d1ea.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/28688510/160430621-76e33cc2-9c5f-4d36-8ee3-5cc0431f9f1d.png">

```c++
class Solution {
public:
    Node* connect(Node* root) {
        if (root == nullptr) {
            return root;
        }
        
        // 从根节点开始
        Node* leftmost = root;
        
        while (leftmost->left != nullptr) {
            
            // 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针
            Node* head = leftmost;
            
            while (head != nullptr) {
                
                // CONNECTION 1
                head->left->next = head->right;
                
                // CONNECTION 2
                if (head->next != nullptr) {
                    head->right->next = head->next->left;
                }
                
                // 指针向后移动
                head = head->next;
            }
            
            // 去下一层的最左的节点
            leftmost = leftmost->left;
        }
        
        return root;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160431381-3f6630e1-41a8-40f4-82d2-0b8be3a39aa3.png">

刚说完变态，变态就来了，要求用O(1)做，且不是完美二叉树，换言之，就是不一定每个节点都有左右孩子。

代码跟上一题基本一样，唯一不同就是需要一个变量去记录下一层起始点，因为不一定是最左边的做孩子了（不一定有左孩子）

```c++
class Solution {
public:
    void handle(Node* &last, Node* &p, Node* &nextStart) {
        if (last) {
            last->next = p;
        } 
        // 记录第一个连接的节点
        if (!nextStart) {
            nextStart = p;
        }
        last = p;
    }

    Node* connect(Node* root) {
        if (!root) {
            return nullptr;
        }
        Node *start = root;
        while (start) {
            Node *last = nullptr, *nextStart = nullptr;
            for (Node *p = start; p != nullptr; p = p->next) {
                if (p->left) {
                    handle(last, p->left, nextStart);
                }
                if (p->right) {
                    handle(last, p->right, nextStart);
                }
            }
            start = nextStart;
        }
        return root;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160436279-ddef8e91-30a9-4089-9dd7-f5523c772476.png">

DFS递归做法，分别求两边的路径和（当计算出负数时置为0），最大路径和为左路径和+右路径和+当前节点值

```c++
class Solution {
private:
    int res = INT_MIN;
public:
    // 以root为起始点的最大路径和
    int dfs(TreeNode* root) {
        if (!root) return 0;
        int left = max(dfs(root->left), 0);
        int right = max(dfs(root->right), 0);
        res = max(res, left + right + root->val);
        // 因为root为起点，最终只取大的一边的路径和
        return max(left, right) + root->val;
    }

    int maxPathSum(TreeNode* root) {
        dfs(root);
        return res;
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160439942-8f4179d8-e85e-4cad-a74d-5b0a6d1e3aec.png">

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160440098-d5c7bef6-689b-428f-a754-47e7b1073916.png">

把标记过的字母 O 修改为字母 A

1. DFS

```c++
class Solution {
public:
    int n, m;

    void dfs(vector<vector<char>>& board, int x, int y) {
        if (x < 0 || x >= n || y < 0 || y >= m || board[x][y] != 'O') {
            return;
        }
        board[x][y] = 'A';
        dfs(board, x + 1, y);
        dfs(board, x - 1, y);
        dfs(board, x, y + 1);
        dfs(board, x, y - 1);
    }

    void solve(vector<vector<char>>& board) {
        n = board.size();
        if (n == 0) {
            return;
        }
        m = board[0].size();
        // 外围的O全都遍历
        for (int i = 0; i < n; i++) {
            dfs(board, i, 0);
            dfs(board, i, m - 1);
        }
        // 外围的O全都遍历
        for (int i = 1; i < m - 1; i++) {
            dfs(board, 0, i);
            dfs(board, n - 1, i);
        }
        
        // 遍历过的A设回O
        // 没遍历到的O改成X，说明是被包围住的O
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }
};
```

2. BFS

```c++
class Solution {
public:
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    void solve(vector<vector<char>>& board) {
        int n = board.size();
        if (n == 0) {
            return;
        }
        int m = board[0].size();
        queue<pair<int, int>> que;
        // 外围的O都遍历（入队）
        for (int i = 0; i < n; i++) {
            if (board[i][0] == 'O') {
                que.emplace(i, 0);
                board[i][0] = 'A';
            }
            if (board[i][m - 1] == 'O') {
                que.emplace(i, m - 1);
                board[i][m - 1] = 'A';
            }
        }
        for (int i = 1; i < m - 1; i++) {
            if (board[0][i] == 'O') {
                que.emplace(0, i);
                board[0][i] = 'A';
            }
            if (board[n - 1][i] == 'O') {
                que.emplace(n - 1, i);
                board[n - 1][i] = 'A';
            }
        }
        while (!que.empty()) {
            int x = que.front().first, y = que.front().second;
            que.pop();
            // 向上下左右四个方向遍历
            for (int i = 0; i < 4; i++) {
                int mx = x + dx[i], my = y + dy[i];
                if (mx < 0 || my < 0 || mx >= n || my >= m || board[mx][my] != 'O') {
                    continue;
                }
                que.emplace(mx, my);
                board[mx][my] = 'A';
            }
        }
        // 遍历过的A设回O
        // 没遍历到的O改成X，说明是被包围住的O
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }
};
```

------

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160442305-63189f8b-94a5-44b5-8f97-9e168cf2b0e9.png">

一边遍历要一边构造新的节点，两种方法都需要借助哈希表记录被克隆过的节点来避免陷入死循环

1. DFS

```c++
class Solution {
public:
    unordered_map<Node*, Node*> visited;
    Node* cloneGraph(Node* node) {
        if (node == nullptr) {
            return node;
        }

        // 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
        if (visited.find(node) != visited.end()) {
            return visited[node];
        }

        // 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
        Node* cloneNode = new Node(node->val);
        // 哈希表存储
        visited[node] = cloneNode;

        // 遍历该节点的邻居并更新克隆节点的邻居列表
        for (auto& neighbor: node->neighbors) {
            cloneNode->neighbors.emplace_back(cloneGraph(neighbor));
        }
        return cloneNode;
    }
};
```

2. BFS

```c++
class Solution {
public:
    Node* cloneGraph(Node* node) {
        if (node == nullptr) {
            return node;
        }

        unordered_map<Node*, Node*> visited;

        // 将题目给定的节点添加到队列
        queue<Node*> Q;
        Q.push(node);
        // 克隆第一个节点并存储到哈希表中
        visited[node] = new Node(node->val);

        // 广度优先搜索
        while (!Q.empty()) {
            // 取出队列的头节点
            auto n = Q.front();
            Q.pop();
            // 遍历该节点的邻居
            for (auto& neighbor: n->neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    // 如果没有被访问过，就克隆并存储在哈希表中
                    visited[neighbor] = new Node(neighbor->val);
                    // 将邻居节点加入队列中
                    Q.push(neighbor);
                }
                // 更新当前节点的邻居列表
                visited[n]->neighbors.emplace_back(visited[neighbor]);
            }
        }

        return visited[node];
    }
};
```

## KMP

<img width="641" alt="image" src="https://user-images.githubusercontent.com/28688510/160673648-495b95e9-1c82-4ca0-82b7-99561238c97f.png">

这里有个[链接](https://www.zhihu.com/question/21923021/answer/281346746)讲的非常好，很详细，但是构建next数组时有循环条件有点错误，需要注意一下

```c++
class Solution {
public:
    int* buildNext(string s) {
        int m = s.size();
        int* next = new int[m];
        next[0] = -1;

        int i = 0, j = -1;
        // 注意这里千万不能用i < m，因为先让i++ j++再next[i] = j，会导致越界
        while (i < m - 1) {
            if (j == -1 || s[i] == s[j]) {
                i++;
                j++;
                next[i] = j;
            }
            else {
                j = next[j];
            }
        }
        return next;
    }
    int strStr(string haystack, string needle) {
        int n = haystack.size();
        int m = needle.size();
        if (m == 0) {
            return 0;
        }

        int i = 0, j = 0;

        int* next = buildNext(needle);

        while (i < n && j < m) {
            if (j == -1 || haystack[i] == needle[j]) {
                i++;
                j++;
            }
            else {
                j = next[j];
            }
        }

        if (j == m) {
            return i - j;
        }
        return -1;
    }
};
```

## 矩阵相关

<img width="518" alt="image" src="https://user-images.githubusercontent.com/28688510/160884535-d792210d-b94d-40b3-930f-d62236b036ab.png">

<img width="767" alt="image" src="https://user-images.githubusercontent.com/28688510/160885023-7552a150-9c2d-42f5-8cce-c9fcacf4eabd.png">

```c++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                swap(matrix[i][j], matrix[n - i - 1][j]);
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};
```

## 括号相关

一般是用来判断括号的合法性或者删除无效的括号，可以用栈来做，也可以直接通过数括号的个数来

<img width="538" alt="image" src="https://user-images.githubusercontent.com/28688510/160886403-b677a88a-697b-4788-b7f9-484489900c71.png">

这个题因为括号不止一种，所以只能通过栈来一个个确认

```c++
class Solution {
public:
    bool isValid(string s) {
        int n = s.size();
        if (n % 2 == 1) {
            return false;
        }
        
        // 这个字典用来区分左右括号的
        unordered_map<char, char> pairs = {
            {')', '('},
            {']', '['},
            {'}', '{'}
        };
        stack<char> stk;
        for (char ch: s) {
            // 如果是右括号
            if (pairs.count(ch)) {
                if (stk.empty() || stk.top() != pairs[ch]) {
                    return false;
                }
                stk.pop();
            }
            // 如果是左括号
            else {
                stk.push(ch);
            }
        }
        return stk.empty();
    }
};
```

------

<img width="539" alt="image" src="https://user-images.githubusercontent.com/28688510/160887195-10fddd39-20e9-46ae-8f86-4dced33a95ab.png">

这题虽然是括号，但是穷举所有可能的情况，这种需要用到回溯的方法，回溯也是需要用到递归的，**回溯的一个需要注意的点是，在回溯前做的操作，在回溯后要撤销掉，回溯时要把所有这一步能做的操作都做一遍**

```c++
class Solution {
    void backtrack(vector<string>& ans, string& cur, int open, int close, int n) {
        if (cur.size() == n * 2) {
            ans.push_back(cur);
            return;
        }
        
        // *** 这里只能进行两种操作，一种加(，一种加)，都要尝试 ***
        
        // 当左括号小于n时才能进行
        if (open < n) {
            cur.push_back('(');
            backtrack(ans, cur, open + 1, close, n);
            cur.pop_back();
        }
        // 同样，右括号个数应该小于左括号个数
        if (close < open) {
            cur.push_back(')');
            backtrack(ans, cur, open, close + 1, n);
            cur.pop_back();
        }
        
        // *** 尝试结束 ***
    }
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        string current;
        backtrack(result, current, 0, 0, n);
        return result;
    }
};
```

------

<img width="526" alt="image" src="https://user-images.githubusercontent.com/28688510/160892452-2699cf20-bce5-4573-a51b-5d0a37554d28.png">

看到最长两个字，应当想到是否可以采用动态规划dp来做，当然也可以用栈来做

1. dp

<img width="766" alt="image" src="https://user-images.githubusercontent.com/28688510/160892965-56c511f8-51d0-44b7-96ab-766b12c3255b.png">

```c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxans = 0, n = s.length();
        vector<int> dp(n, 0);
        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                if (s[i - 1] == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = max(maxans, dp[i]);
            }
        }
        return maxans;
    }
};
```

2. 栈

```c++
class Solution {
public:
    int longestValidParentheses(string s) {
        int maxans = 0;
        stack<int> stk;
        stk.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s[i] == '(') {
                stk.push(i);
            } else {
                stk.pop();
                if (stk.empty()) {
                    stk.push(i);
                } else {
                    maxans = max(maxans, i - stk.top());
                }
            }
        }
        return maxans;
    }
};
```
