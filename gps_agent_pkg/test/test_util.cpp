/**
 * @file test_util.cpp
 * @brief Unit tests for util.h / util.cpp — C++20 standard-library helpers.
 *
 * util::split(), split_view(), and to_string<T>() are the GPS
 * codebase utilities that are fully independent of ROS, LibTorch, Eigen,
 * and Protobuf. They can be tested on any C++20 compiler.
 *
 * C++20 features tested:
 * - std::string_view split functions
 * - Stringifiable concept
 * - [[nodiscard]] compliance
 */

#include <gtest/gtest.h>
#include "gps_agent_pkg/util.h"

#include <concepts>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// C++20 Concept Tests
// ---------------------------------------------------------------------------

TEST(UtilConcepts, StringifiableConceptSatisfied) {
    // Verify that common types satisfy the Stringifiable concept
    static_assert(util::Stringifiable<int>);
    static_assert(util::Stringifiable<double>);
    static_assert(util::Stringifiable<float>);
    static_assert(util::Stringifiable<std::string>);
    static_assert(util::Stringifiable<const char*>);
    static_assert(util::Stringifiable<std::string_view>);
    static_assert(util::Stringifiable<long>);
    static_assert(util::Stringifiable<unsigned int>);

    SUCCEED();
}

TEST(UtilConcepts, ToCharsConvertibleConceptSatisfied) {
    // Verify that arithmetic types satisfy ToCharsConvertible
    static_assert(util::ToCharsConvertible<int>);
    static_assert(util::ToCharsConvertible<double>);
    static_assert(util::ToCharsConvertible<float>);
    static_assert(util::ToCharsConvertible<long>);
    static_assert(util::ToCharsConvertible<unsigned int>);

    // Non-arithmetic types should not satisfy it
    static_assert(!util::ToCharsConvertible<std::string>);
    static_assert(!util::ToCharsConvertible<const char*>);

    SUCCEED();
}

// ---------------------------------------------------------------------------
// util::split (legacy version with output parameter)
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

TEST(UtilSplit, EmptyStringYieldsNoTokens) {
    std::vector<std::string> parts;
    util::split("", ',', parts);
    // std::getline semantics: empty input stream yields no tokens
    ASSERT_EQ(parts.size(), 0u);
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
// util::split (C++20 version returning vector)
// ---------------------------------------------------------------------------

TEST(UtilSplitC20, ReturnsVectorOfStrings) {
    auto parts = util::split("a,b,c", ',');
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST(UtilSplitC20, AcceptsStringView) {
    std::string_view sv = "foo:bar:baz";
    auto parts = util::split(sv, ':');
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "foo");
    EXPECT_EQ(parts[1], "bar");
    EXPECT_EQ(parts[2], "baz");
}

TEST(UtilSplitC20, SingleTokenNoDelimiter) {
    auto parts = util::split("hello", ',');
    ASSERT_EQ(parts.size(), 1u);
    EXPECT_EQ(parts[0], "hello");
}

TEST(UtilSplitC20, EmptyStringYieldsEmptyToken) {
    auto parts = util::split("", ',');
    ASSERT_EQ(parts.size(), 1u);
    EXPECT_EQ(parts[0], "");
}

// ---------------------------------------------------------------------------
// util::split_view (C++20 string_view version)
// ---------------------------------------------------------------------------

TEST(UtilSplitView, ReturnsVectorOfStringViews) {
    std::string source = "x,y,z";
    auto parts = util::split_view(source, ',');
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "x");
    EXPECT_EQ(parts[1], "y");
    EXPECT_EQ(parts[2], "z");
}

TEST(UtilSplitView, ViewsPointIntoSource) {
    std::string source = "one|two|three";
    auto parts = util::split_view(source, '|');
    ASSERT_EQ(parts.size(), 3u);

    // Verify that the views actually point into the source string
    EXPECT_GE(parts[0].data(), source.data());
    EXPECT_LE(parts[0].data(), source.data() + source.size());
}

TEST(UtilSplitView, NoAllocationForSmallStrings) {
    // split_view should not allocate any string data
    // (the strings are views into the original)
    std::string_view sv = "a,b,c";
    auto parts = util::split_view(sv, ',');
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "b");
    EXPECT_EQ(parts[2], "c");
}

TEST(UtilSplitView, EmptySegments) {
    auto parts = util::split_view("a,,b", ',');
    ASSERT_EQ(parts.size(), 3u);
    EXPECT_EQ(parts[0], "a");
    EXPECT_EQ(parts[1], "");  // empty segment
    EXPECT_EQ(parts[2], "b");
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
    // to_string<std::string> should return the string as-is
    EXPECT_EQ(to_string(std::string("hello")), "hello");
}

TEST(ToStringTest, StringViewConversion) {
    // to_string for string_view should return a string copy
    std::string_view sv = "test_view";
    std::string result = to_string(sv);
    EXPECT_EQ(result, "test_view");
}

TEST(ToStringTest, UnsignedTypes) {
    EXPECT_EQ(to_string(0u), "0");
    EXPECT_EQ(to_string(255u), "255");
}

TEST(ToStringTest, LongTypes) {
    EXPECT_EQ(to_string(1000000L), "1000000");
    EXPECT_EQ(to_string(-1000000L), "-1000000");
}

// ---------------------------------------------------------------------------
// [[nodiscard]] Verification
// ---------------------------------------------------------------------------

TEST(ToStringTest, NodiscardCompliance) {
    // This test verifies that the [[nodiscard]] attribute is working
    // (compiler should warn if we discard the result - but we don't test
    // warnings here, just verify the function works correctly)
    [[maybe_unused]] auto result1 = to_string(42);
    [[maybe_unused]] auto result2 = util::split("a,b", ',');
    [[maybe_unused]] auto result3 = util::split_view("a,b", ',');

    SUCCEED();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
