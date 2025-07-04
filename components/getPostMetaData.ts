import fs from "fs";
import matter from "gray-matter";
import { PostMetaData } from "@/components/PostMetaData";
// this is where we store all posts
const folder = "posts/";

const readFrontmatter = (fileName: string): PostMetaData => {
  const content = fs.readFileSync(`${folder}/${fileName}`);
  const frontmatter = matter(content);
  return {
    title: frontmatter.data.title,
    date: frontmatter.data.date,
    subtitle: frontmatter.data.subtitle,
    slug: fileName.replace(".md", ""),
  };
};

const getPostMetaData = (): PostMetaData[] => {
  const files = fs.readdirSync(folder);
  const markDownFiles = files.filter((file) => file.endsWith(".md"));
  const meta = markDownFiles.map((file) => readFrontmatter(file));
  return meta;
};

export default getPostMetaData;
