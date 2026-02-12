/**
 * test_util.cpp
 *
 * Unit tests for util.h / util.cpp — pure C++ standard-library helpers.
 *
 * util::split() and to_string<T>() are the only utilities in the GPS
 * codebase that are fully independent of ROS, LibTorch, Eigen, and Protobuf.
 * They can be tested on any C++17 compiler with no additional dependencies.
 */

#include <gtest/gtest.h>
#include "gps_agent_pkg/util.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// util::split
// ---------------------------------------------------------------------------

TEST(UtilSplit, CommaSeparatedThreeTokens) {
    std::vector<std::string> parts;
    util::split("a,b,c", ',', parts);
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST(UtilSplit, SpaceDelimiter) {
    std::vector<std::string> parts;
    util::split("foo bar baz", ' ', parts);
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "foo");
    EXPECT_EQ(parts[1], "bar");
    EXPECT_EQ(parts[2], "baz");
}

TEST(UtilSplit, SingleToken) {
    std::vector<std::string> parts;
    util::split("hello", ',', parts);
    ASSERT_EQ(parts.size(), 1u);
    EXPECT_EQ(parts[0], "hello");
}

TEST(UtilSplit, EmptyStringYieldsOneEmptyToken) {
    std::vector<std::string> parts;
    util::split("", ',', parts);
    // std::getline semantics: empty input → one empty token
    ASSERT_EQ(parts.size(), 1u);
    EXPECT_EQ(parts[0], "");
}

TEST(UtilSplit, TwoTokens) {
    std::vector<std::string> parts;
    util::split("key=value", '=', parts);
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0], "key");
    EXPECT_EQ(parts[1], "value");
}

TEST(UtilSplit, AppendsToParts) {
    // split appends to the existing vector — calling twice accumulates
    std::vector<std::string> parts;
    util::split("a,b", ',', parts);
    util::split("c,d", ',', parts);
    ASSERT_EQ(parts.size(), 4u);
    EXPECT_EQ(parts[2], "c");
    EXPECT_EQ(parts[3], "d");
}

// ---------------------------------------------------------------------------
// to_string<T>
// ---------------------------------------------------------------------------

TEST(ToStringTest, Zero) {
    EXPECT_EQ(to_string(0), "0");
}

TEST(ToStringTest, PositiveInt) {
    EXPECT_EQ(to_string(42), "42");
}

TEST(ToStringTest, NegativeInt) {
    EXPECT_EQ(to_string(-7), "-7");
}

TEST(ToStringTest, FloatNonEmpty) {
    std::string s = to_string(3.14f);
    EXPECT_FALSE(s.empty());
}

TEST(ToStringTest, DoubleNonEmpty) {
    std::string s = to_string(2.718);
    EXPECT_FALSE(s.empty());
}

TEST(ToStringTest, StringPassthrough) {
    // to_string<std::string> should stream the string as-is
    EXPECT_EQ(to_string(std::string("hello")), "hello");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
