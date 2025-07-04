import fs from "fs";
// import Markdown from "markdown-to-jsx";
import Markdown from "react-markdown";
import matter from "gray-matter";
import getPostMetaData from "@/components/getPostMetaData";

import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css"; // `rehype-katex` does not import the CSS for you

const getPageContent = (slug: string) => {
  const folder = "posts";
  const file = `${folder}/${slug}.md`;
  const content = fs.readFileSync(file, "utf-8");
  const frontmatter = matter(content);
  return frontmatter;
};

export const generateStaticParams = async () => {
  const posts = getPostMetaData();
  return posts.map((post) => ({
    slug: post.slug,
  }));
};

const PostPage = async (props: any) => {
  const { slug } = await props.params;
  const post = getPageContent(slug);
  return (
    <div>
      <h2 className="text-2xl text-bold text-violet-400">{post.data.title}</h2>
      <article className="prose prose-base">
        <Markdown
          remarkPlugins={[remarkMath, remarkGfm]}
          rehypePlugins={[rehypeKatex]}
        >
          {post.content}
        </Markdown>
      </article>
    </div>
  );
};

export default PostPage;
