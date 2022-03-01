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
