import fs from "fs";
import matter from "gray-matter";
import { parseISO } from "date-fns";
import { PostMetaData } from "@/components/PostMetaData";
// this is where we store all posts
const folder = "posts/";

const readFrontmatter = (fileName: string): PostMetaData => {
  const content = fs.readFileSync(`${folder}/${fileName}`);
  const frontmatter = matter(content);
  return {
    title: frontmatter.data.title,
    date: parseISO(frontmatter.data.date),
    subtitle: frontmatter.data.subtitle,
    slug: fileName.replace(".md", ""),
    published: frontmatter.data.published,
  };
};

const getPostMetaData = (): PostMetaData[] => {
  const files = fs.readdirSync(folder);
  const markDownFiles = files.filter((file) => file.endsWith(".md"));
  const meta = markDownFiles.map((file) => readFrontmatter(file));
  const filteredPosts = meta.filter((data) => data.published);
  const sortedPosts = filteredPosts.sort(
    (a, b) => b.date.getTime() - a.date.getTime()
  );
  return sortedPosts;
};

export default getPostMetaData;
